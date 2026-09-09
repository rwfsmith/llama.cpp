#include "llama.h"

int llama_cli(int argc, char ** argv);

int main(int argc, char ** argv) {
    const int result = llama_cli(argc, argv);
    llama_backend_free();
    return result;
}
