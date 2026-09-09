#include "llama.h"

int llama_fit_params(int argc, char ** argv);

int main(int argc, char ** argv) {
    const int result = llama_fit_params(argc, argv);
    llama_backend_free();
    return result;
}
