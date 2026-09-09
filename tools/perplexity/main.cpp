#include "llama.h"

int llama_perplexity(int argc, char ** argv);

int main(int argc, char ** argv) {
    const int result = llama_perplexity(argc, argv);
    llama_backend_free();
    return result;
}
