#include "llama.h"

int llama_batched_bench(int argc, char ** argv);

int main(int argc, char ** argv) {
    const int result = llama_batched_bench(argc, argv);
    llama_backend_free();
    return result;
}
