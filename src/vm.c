#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PUSH_CONST 0x01
#define MATMUL 0x02
#define ADD 0x03
#define SIGMOID 0x04
#define ODE_SOLVE 0x05
#define RETURN 0xFF

/* Structure for VM constants */
typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} Matrix;

/* VM state */
typedef struct {
    float *hidden_state;
    size_t hidden_size;
} VMState;

/* Allocate a matrix */
Matrix *matrix_create(size_t rows, size_t cols) {
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    m->data = (float *)calloc(rows * cols, sizeof(float));
    m->rows = rows;
    m->cols = cols;
    return m;
}

/* Free a matrix */
void matrix_free(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

/* Matrix multiplication: C = A * B */
Matrix *matrix_multiply(const Matrix *A, const Matrix *B) {
    if (A->cols != B->rows) return NULL;
    Matrix *C = matrix_create(A->rows, B->cols);
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C->data[i * C->cols + j] = sum;
        }
    }
    return C;
}

/* Matrix addition: C = A + B (B is broadcasted if scalar-like) */
Matrix *matrix_add(const Matrix *A, const Matrix *B) {
    Matrix *C = matrix_create(A->rows, A->cols);
    for (size_t i = 0; i < A->rows * A->cols; i++) {
        float b_val = (B->rows * B->cols == 1) ? B->data[0] : B->data[i];
        C->data[i] = A->data[i] + b_val;
    }
    return C;
}

/* Sigmoid activation */
Matrix *matrix_sigmoid(const Matrix *A) {
    Matrix *C = matrix_create(A->rows, A->cols);
    for (size_t i = 0; i < A->rows * A->cols; i++) {
        float x = A->data[i];
        x = fminf(fmaxf(x, -100.0f), 100.0f); /* Tighter clipping */
        if (x > 100.0f) {
            C->data[i] = 1.0f; /* Avoid computing exp for large x */
        } else if (x < -100.0f) {
            C->data[i] = 0.0f; /* Avoid computing exp for very negative x */
        } else {
            C->data[i] = 1.0f / (1.0f + expf(-x));
        }
    }
    return C;
}

/* Element-wise division: C = A / B (B is broadcasted) */
Matrix *matrix_divide(const Matrix *A, const Matrix *B) {
    Matrix *C = matrix_create(A->rows, A->cols);
    for (size_t i = 0; i < A->rows * A->cols; i++) {
        float b_val = (B->rows * B->cols == 1) ? B->data[0] : B->data[i];
        if (b_val == 0.0f) return NULL;
        C->data[i] = A->data[i] / b_val;
    }
    return C;
}

/* Element-wise multiplication: C = A * B */
Matrix *matrix_multiply_elementwise(const Matrix *A, const Matrix *B) {
    Matrix *C = matrix_create(A->rows, A->cols);
    for (size_t i = 0; i < A->rows * A->cols; i++) {
        float b_val = (B->rows * B->cols == 1) ? B->data[0] : B->data[i];
        C->data[i] = A->data[i] * b_val;
    }
    return C;
}

/* Concatenate matrices A and B horizontally */
Matrix *matrix_concat(const Matrix *A, const Matrix *B) {
    if (A->rows != B->rows) return NULL;
    Matrix *C = matrix_create(A->rows, A->cols + B->cols);
    for (size_t i = 0; i < A->rows; i++) {
        memcpy(C->data + i * C->cols, A->data + i * A->cols, A->cols * sizeof(float));
        memcpy(C->data + i * C->cols + A->cols, B->data + i * B->cols, B->cols * sizeof(float));
    }
    return C;
}

/* LTC ODE solver: dh/dt = (-h + (W @ combined + b) * gate) / tau */
Matrix *ltc_ode_solver(const Matrix *h, const Matrix *x, const Matrix *consts[], size_t const_indices[]) {
    size_t w_gate_idx = const_indices[0];
    size_t b_gate_idx = const_indices[1];
    size_t w_idx = const_indices[2];
    size_t b_idx = const_indices[3];
    size_t tau_idx = const_indices[4];

    /* combined = [x, h] */
    Matrix *combined = matrix_concat(x, h);
    if (!combined) return NULL;

    /* gate = sigmoid(W_gate @ combined + b_gate) */
    Matrix *gate_intermediate = matrix_multiply(combined, consts[w_gate_idx]);
    Matrix *gate = gate_intermediate ? matrix_add(gate_intermediate, consts[b_gate_idx]) : NULL;
    Matrix *gate_sig = gate ? matrix_sigmoid(gate) : NULL;
    matrix_free(gate_intermediate);
    matrix_free(gate);
    matrix_free(combined);
    if (!gate_sig) return NULL;

    /* W @ combined + b */
    combined = matrix_concat(x, h);
    Matrix *w_out = matrix_multiply(combined, consts[w_idx]);
    Matrix *w_out_b = w_out ? matrix_add(w_out, consts[b_idx]) : NULL;
    matrix_free(w_out);
    matrix_free(combined);
    if (!w_out_b) {
        matrix_free(gate_sig);
        return NULL;
    }

    /* (W @ combined + b) * gate */
    Matrix *gated = matrix_multiply_elementwise(w_out_b, gate_sig);
    matrix_free(w_out_b);
    matrix_free(gate_sig);
    if (!gated) return NULL;

    /* -h + gated */
    Matrix *h_neg = matrix_create(h->rows, h->cols);
    for (size_t i = 0; i < h->rows * h->cols; i++) {
        h_neg->data[i] = -h->data[i];
    }
    Matrix *dhdt = matrix_add(h_neg, gated);
    matrix_free(h_neg);
    matrix_free(gated);
    if (!dhdt) return NULL;

    /* (-h + gated) / tau */
    Matrix *result = matrix_divide(dhdt, consts[tau_idx]);
    matrix_free(dhdt);
    return result;
}

/* Euler ODE solver */
Matrix *euler_solve(Matrix *(*ode_func)(const Matrix *, const Matrix *, const Matrix *[], size_t[]),
                    const Matrix *h0, const Matrix *x, const float *t_eval, size_t t_eval_len,
                    const Matrix *consts[], size_t const_indices[]) {
    Matrix *h = matrix_create(h0->rows, h0->cols);
    memcpy(h->data, h0->data, h0->rows * h0->cols * sizeof(float));

    for (size_t i = 0; i < t_eval_len - 1; i++) {
        float dt = t_eval[i + 1] - t_eval[i];
        Matrix *dh = ode_func(h, x, consts, const_indices);
        if (!dh) {
            matrix_free(h);
            return NULL;
        }
        for (size_t j = 0; j < h->rows * h->cols; j++) {
            h->data[j] += dh->data[j] * dt;
        }
        matrix_free(dh);
    }
    return h;
}

/* Check for NaN or Inf */
int is_invalid_float(float x) {
    return isnan(x) || isinf(x);
}

/* Load bytecode from file */
unsigned char *load_bytecode(const char *filename, size_t *length) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *bytecode = (unsigned char *)malloc(*length);
    fread(bytecode, 1, *length, f);
    fclose(f);
    return bytecode;
}

/* Main VM execution function */
float vm_run(float x_input, float x_mean, float x_std, float y_mean, float y_std,
             float *const_data[], size_t const_rows[], size_t const_cols[], size_t const_count,
             const char *bytecode_file, const float *t_eval, size_t t_eval_len) {
    /* Normalize input */
    float x_norm = (x_input - x_mean) / x_std;

    /* Create constants */
    Matrix **consts = (Matrix **)malloc(const_count * sizeof(Matrix *));
    for (size_t i = 0; i < const_count; i++) {
        consts[i] = matrix_create(const_rows[i], const_cols[i]);
        memcpy(consts[i]->data, const_data[i], const_rows[i] * const_cols[i] * sizeof(float));
    }

    /* Initialize stack with input: [[x_norm]] */
    Matrix *stack[16]; /* Fixed-size stack for simplicity */
    size_t stack_top = 0;
    stack[stack_top] = matrix_create(1, 1);
    stack[stack_top]->data[0] = x_norm;
    stack_top++;

    /* Load bytecode */
    size_t bytecode_len;
    unsigned char *bytecode = load_bytecode(bytecode_file, &bytecode_len);
    if (!bytecode) {
        for (size_t i = 0; i < stack_top; i++) matrix_free(stack[i]);
        for (size_t i = 0; i < const_count; i++) matrix_free(consts[i]);
        free(consts);
        return NAN;
    }

    size_t ip = 0;
    while (ip < bytecode_len) {
        unsigned char op = bytecode[ip++];

        if (op == PUSH_CONST) {
            size_t idx = bytecode[ip++];
            if (idx >= const_count || stack_top >= 16) goto error;
            stack[stack_top++] = matrix_create(consts[idx]->rows, consts[idx]->cols);
            memcpy(stack[stack_top-1]->data, consts[idx]->data, consts[idx]->rows * consts[idx]->cols * sizeof(float));
        }
        else if (op == MATMUL) {
            if (stack_top < 2) goto error;
            Matrix *w = stack[--stack_top];
            Matrix *x = stack[--stack_top];
            Matrix *result = matrix_multiply(x, w);
            matrix_free(w);
            matrix_free(x);
            if (!result) goto error;
            stack[stack_top++] = result;
        }
        else if (op == ADD) {
            if (stack_top < 2) goto error;
            Matrix *b = stack[--stack_top];
            Matrix *x = stack[--stack_top];
            Matrix *result = matrix_add(x, b);
            matrix_free(b);
            matrix_free(x);
            if (!result) goto error;
            stack[stack_top++] = result;
        }
        else if (op == SIGMOID) {
            if (stack_top < 1) goto error;
            Matrix *x = stack[--stack_top];
            Matrix *result = matrix_sigmoid(x);
            matrix_free(x);
            if (!result) goto error;
            stack[stack_top++] = result;
        }
        else if (op == ODE_SOLVE) {
            if (stack_top < 1 || ip + 7 > bytecode_len) goto error;
            size_t const_indices[5];
            for (size_t i = 0; i < 5; i++) {
                const_indices[i] = bytecode[ip + i];
                if (const_indices[i] >= const_count) goto error;
            }
            size_t w_out_idx = bytecode[ip + 5];
            size_t b_out_idx = bytecode[ip + 6];
            if (w_out_idx >= const_count || b_out_idx >= const_count) goto error;
            ip += 7;

            Matrix *x = stack[stack_top - 1];
            Matrix *h0 = matrix_create(1, consts[const_indices[0]]->cols);
            Matrix *h_final = euler_solve(ltc_ode_solver, h0, x, t_eval, t_eval_len, (const Matrix **)consts, const_indices);
            matrix_free(h0);
            if (!h_final) goto error;

            Matrix *out = matrix_multiply(h_final, consts[w_out_idx]);
            Matrix *out_final = out ? matrix_add(out, consts[b_out_idx]) : NULL;
            matrix_free(out);
            matrix_free(h_final);
            if (!out_final) goto error;

            stack[stack_top++] = out_final;
        }
        else if (op == RETURN) {
            if (stack_top < 1) goto error;
            Matrix *result = stack[--stack_top];
            float output = result->data[0];
            matrix_free(result);

            /* Clean up */
            for (size_t i = 0; i < stack_top; i++) matrix_free(stack[i]);
            for (size_t i = 0; i < const_count; i++) matrix_free(consts[i]);
            free(consts);
            free(bytecode);

            if (is_invalid_float(output)) return NAN;
            return output * y_std + y_mean;
        }
        else {
            goto error;
        }
    }

error:
    for (size_t i = 0; i < stack_top; i++) matrix_free(stack[i]);
    for (size_t i = 0; i < const_count; i++) matrix_free(consts[i]);
    free(consts);
    free(bytecode);
    return NAN;
}