//
// Created by jonossar on 4/28/24.
//

#ifndef IMG1_KERNELCOMMONS_CUH
#define IMG1_KERNELCOMMONS_CUH
#define thisStream (*(streams.at(std::this_thread::get_id())))
inline std::map<std::thread::id, cudaStream_t*> streams{};
#define ERC(ans) {gpuAssertHost(ans, __FILE__,__LINE__);}
#define errcGpu(ans){gpuAssertDevice(ans,__FILE__,__LINE__);}

__device__ inline void gpuAssertDevice(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("%s::%s::%d\n", cudaGetErrorString(code), file, line);
    }
}
__host__ inline void gpuAssertHost(cudaError_t code, const char *file, int line, bool ex = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "%s::%s::%d\n", cudaGetErrorString(code), file, line);
        if(ex) abort();
    }

}
__device__
static inline
uint8_t
atomicCAS(uint8_t *const address,
          uint8_t const compare,
          uint8_t const value) {
    // Determine where in a byte-aligned 32-bit range our address of 8 bits occurs.
    uint8_t const longAddressModulo = reinterpret_cast< size_t >( address ) & 0x3;
    // Determine the base address of the byte-aligned 32-bit range that contains our address of 8 bits.
    uint32_t *const baseAddress = reinterpret_cast< uint32_t * >( address - longAddressModulo );
    uint32_t constexpr byteSelection[] = {0x3214, 0x3240, 0x3410,
                                          0x4210}; // The byte position we work on is '4'.
    uint32_t const byteSelector = byteSelection[longAddressModulo];
    uint32_t const longCompare = compare;
    uint32_t const longValue = value;
    uint32_t longOldValue = *baseAddress;
    uint32_t longAssumed;
    uint8_t oldValue;
    do {
        // Select bytes from the old value and new value to construct a 32-bit value to use.
        uint32_t const replacement = __byte_perm(longOldValue, longValue, byteSelector);
        uint32_t const comparison = __byte_perm(longOldValue, longCompare, byteSelector);

        longAssumed = longOldValue;
        // Use 32-bit atomicCAS() to try and set the 8-bits we care about.
        longOldValue = ::atomicCAS(baseAddress, comparison, replacement);
        // Grab the 8-bit portion we care about from the old value row address.
        oldValue = (longOldValue >> (8 * longAddressModulo)) & 0xFF;
    } while (compare == oldValue and
             longAssumed != longOldValue); // Repeat until other three 8-bit values stabilize.

    return oldValue;
}
__device__ static inline char atomicAdd(char *address, char val) {
    // offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
    size_t long_address_modulo = (size_t) address & 3;
    // the 32-bit address that overlaps the same memory
    auto *base_address = (unsigned int *) ((char *) address - long_address_modulo);
    // A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
    // The "4" signifies the position where the first byte of the second argument will end up in the output.
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    // for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
    unsigned int selector = selectors[long_address_modulo];
    unsigned int long_old, long_assumed, long_val, replacement;

    long_old = *base_address;

    do {
        long_assumed = long_old;
        // replace bits in long_old that pertain to the char address with those from val
        long_val = __byte_perm(long_old, 0, long_address_modulo) + val;
        replacement = __byte_perm(long_old, long_val, selector);
        long_old = atomicCAS(base_address, long_assumed, replacement);
    } while (long_old != long_assumed);
    return __byte_perm(long_old, 0, long_address_modulo);
}
#endif //IMG1_KERNELCOMMONS_CUH
