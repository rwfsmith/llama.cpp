#include "llama.h"

int llama_completion(int argc, char ** argv);

int main(int argc, char ** argv) {
    const int result = llama_completion(argc, argv);
    llama_backend_free();
    return result;
}
