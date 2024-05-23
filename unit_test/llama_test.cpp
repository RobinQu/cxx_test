//
// Created by RobinQu on 2024/5/21.
//
#include <llama.h>
#include <ggml.h>


static void fn1() {

    // define function
    ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = nullptr
    };
    ggml_context *ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_param(ctx, x);
    ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * x2 = ggml_mul(ctx, x, x);
    ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b); // f(x) = ax^2 + b

    const auto size_used = ggml_used_mem(ctx);
    printf("used = %lu\n", size_used);

    // run function
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f);
    ggml_set_f32(x, 2.0f);
    ggml_set_f32(a, 3.0f);
    ggml_set_f32(b, 4.0f);

    ggml_graph_compute_with_ctx(ctx, gf, 1);

    printf("f = %f\n", ggml_get_f32_1d(f, 0));

}



int main() {

    fn1();

}