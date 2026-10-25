#include "selftest.h"
#include "ops.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Fixed pseudo-random values between -0.5 and 0.5, so every run tests the same numbers.
static void fill_random(float *values, long count, unsigned long long *random_state) {
    for (long i = 0; i < count; i++) {
        *random_state = *random_state * 6364136223846793005ULL + 1;
        values[i] = (float)(*random_state >> 40) / 16777216.0f - 0.5f;
    }
}

int run_selftest(void) {
    int failures = 0;

    float matrix[6] = {1, 2, 3, 4, 5, 6}, ones_vector[3] = {1, 1, 1}, matmul_result[2];
    matmul(matmul_result, ones_vector, matrix, 3, 2);
    if (matmul_result[0] != 6 || matmul_result[1] != 15) { printf("FAIL matmul\n"); failures++; }

    // int8: going to int8 and back should be off by half a step at most, and
    // matmul_q8 should land close to matmul.
    enum { TEST_IN_DIM = 2 * Q8_GROUP, TEST_OUT_DIM = 3, TEST_WEIGHT_COUNT = TEST_IN_DIM * TEST_OUT_DIM };
    float float_weights[TEST_WEIGHT_COUNT], float_input[TEST_IN_DIM];
    float round_trip_weights[TEST_WEIGHT_COUNT];
    float expected_out[TEST_OUT_DIM], int8_out[TEST_OUT_DIM];
    int8_t int8_weights[TEST_WEIGHT_COUNT], int8_input[TEST_IN_DIM];
    float weight_scales[TEST_WEIGHT_COUNT / Q8_GROUP], input_scales[TEST_IN_DIM / Q8_GROUP];
    for (int i = 0; i < TEST_WEIGHT_COUNT; i++) float_weights[i] = sinf(i * 0.37f) * (i % 7 + 1);
    for (int i = 0; i < TEST_IN_DIM; i++) float_input[i] = cosf(i * 0.11f);
    quantize_q8(int8_weights, weight_scales, float_weights, TEST_WEIGHT_COUNT);
    dequantize_q8(round_trip_weights, int8_weights, weight_scales, TEST_WEIGHT_COUNT);
    int round_trip_ok = 1;
    for (int i = 0; i < TEST_WEIGHT_COUNT; i++)
        if (fabsf(round_trip_weights[i] - float_weights[i]) > weight_scales[i / Q8_GROUP] * 0.5f + 1e-6f)
            round_trip_ok = 0;
    if (!round_trip_ok) { printf("FAIL quantize_q8 round trip\n"); failures++; }
    quantize_q8(int8_input, input_scales, float_input, TEST_IN_DIM);
    matmul(expected_out, float_input, float_weights, TEST_IN_DIM, TEST_OUT_DIM);
    matmul_q8(int8_out, int8_input, input_scales, int8_weights, weight_scales, TEST_IN_DIM, TEST_OUT_DIM);
    for (int i = 0; i < TEST_OUT_DIM; i++)
        if (fabsf(int8_out[i] - expected_out[i]) > 0.02f * (fabsf(expected_out[i]) + 1.0f)) {
            printf("FAIL matmul_q8 (row %d: %f vs %f)\n", i, int8_out[i], expected_out[i]); failures++; break;
        }

    // The threaded matmul should give exactly the same numbers as one thread.
    // Check the stories15M sizes, plus a couple that don't split evenly.
    int test_shapes[][2] = {{288, 288}, {288, 768}, {768, 288}, {288, 32000}, {7, 5}, {13, 3}};
    int shape_count = sizeof(test_shapes) / sizeof(test_shapes[0]);
    for (int shape_index = 0; shape_index < shape_count; shape_index++) {
        int input_size = test_shapes[shape_index][0], output_size = test_shapes[shape_index][1];
        float *test_weights = malloc((size_t)input_size * output_size * sizeof(float));
        float *test_input = malloc(input_size * sizeof(float));
        float *single_thread_result = malloc(output_size * sizeof(float));
        float *multi_thread_result = malloc(output_size * sizeof(float));

        unsigned long long random_state = 42;
        fill_random(test_weights, (long)input_size * output_size, &random_state);
        fill_random(test_input, input_size, &random_state);

        matmul(single_thread_result, test_input, test_weights, input_size, output_size);
        for (int thread_count = 2; thread_count <= 8; thread_count *= 2) {
            ops_init(thread_count);
            matmul(multi_thread_result, test_input, test_weights, input_size, output_size);
            ops_shutdown();
            if (memcmp(single_thread_result, multi_thread_result, output_size * sizeof(float)) != 0) {
                printf("FAIL threaded matmul (input_size=%d output_size=%d threads=%d)\n",
                       input_size, output_size, thread_count);
                failures++;
            }
        }
        free(test_weights); free(test_input); free(single_thread_result); free(multi_thread_result);
    }

    float softmax_values[3] = {1, 2, 3};
    softmax(softmax_values, 3);
    float probability_sum = softmax_values[0] + softmax_values[1] + softmax_values[2];
    if (fabsf(probability_sum - 1.0f) > 1e-5f) { printf("FAIL softmax (sum = %f)\n", probability_sum); failures++; }

    float norm_input[2] = {3, 4}, norm_weight[2] = {1, 1}, norm_output[2];
    rmsnorm(norm_output, norm_input, norm_weight, 2);
    float expected = 3.0f / sqrtf(12.5f + 1e-5f);
    if (fabsf(norm_output[0] - expected) > 1e-4f) { printf("FAIL rmsnorm\n"); failures++; }

    printf(failures ? "%d test(s) failed\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
