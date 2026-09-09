#include "llama.h"

int llama_quantize(int argc, char ** argv);

int main(int argc, char ** argv) {
    const int result = llama_quantize(argc, argv);
    llama_backend_free();
    return result;
}
