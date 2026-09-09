#include "llama.h"

int llama_server(int argc, char ** argv);

int main(int argc, char ** argv) {
    const int result = llama_server(argc, argv);
    llama_backend_free();
    return result;
}
