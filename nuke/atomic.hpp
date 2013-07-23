#pragma once

#include <cstddef> // for size_t

#ifndef __host__
#define __host__
#define NUKE_UNDEF_HOST
#endif

#ifndef __device__
#define __device__
#define NUKE_UNDEF_DEVICE
#endif

namespace nuke
{


namespace detail
{


template<bool condition, typename Result = void>
struct enable_if {};


template<typename Result>
struct enable_if<true,Result>
{
  typedef Result type;
};


template<typename Integer32>
__host__ __device__
typename enable_if<
  sizeof(Integer32) == 4
>::type
atomic_store_n(Integer32 *x, Integer32 y)
{
#if defined(__CUDA_ARCH__)
  atomicExch(x, y);
#elif defined(__GNUC__)
  return __atomic_store_n(x, y, __ATOMIC_SEQ_CST);
#elif defined(_MSC_VER)
  InterlockedExchange(x, y); 
#elif defined(__clang__)
  __c11_atomic_store(x, y);
#else
#error "No atomic_store_n implementation."
#endif
}


template<typename Integer64>
__host__ __device__
typename enable_if<
  sizeof(Integer64) == 8
>::type
atomic_store_n(Integer64 *x, Integer64 y)
{
#if defined(__CUDA_ARCH__)
  atomicExch(x, y);
#elif defined(__GNUC__)
  return __atomic_store_n(x, y, __ATOMIC_SEQ_CST);
#elif defined(_MSC_VER)
  InterlockedExchange64(x, y); 
#elif defined(__clang__)
  __c11_atomic_store(x, y);
#else
#error "No atomic_store_n implementation."
#endif
}


template<typename Integer32>
__host__ __device__
typename enable_if<
  sizeof(Integer32) == 4,
  Integer32
>::type
atomic_load_n(const Integer32 *x)
{
#if defined(__CUDA_ARCH__)
  return atomicAdd(const_cast<Integer32*>(x), Integer32(0));
#elif defined(__GNUC__)
  return __atomic_load_n(x, __ATOMIC_SEQ_CST);
#elif defined(_MSC_VER)
  return InterlockedExchangeAdd(x, Integer32(0));
#elif defined(__clang__)
  return __c11_atomic_load(x);
#else
#error "No atomic_load_n implementation."
#endif
}


template<typename Integer64>
__host__ __device__
typename enable_if<
  sizeof(Integer64) == 8,
  Integer64
>::type
atomic_load_n(const Integer64 *x)
{
#if defined(__CUDA_ARCH__)
  return atomicAdd(const_cast<Integer64*>(x), Integer64(0));
#elif defined(__GNUC__)
  return atomic_load_n(x, __ATOMIC_SEQ_CST);
#elif defined(_MSC_VER)
  return InterlockedExchangeAdd(x, Integer64(0));
#elif defined(__clang__)
  return __c11_atomic_load(x);
#else
#error "No atomic_load_n implementation."
#endif
}


// atomic_fetch_op returns the *old* value at x

template<typename Integer32>
__host__ __device__
typename enable_if<
  sizeof(Integer32) == 4,
  Integer32
>::type
atomic_fetch_add(Integer32 *x, Integer32 y)
{
#if defined(__CUDA_ARCH__)
  return atomicAdd(x, y);
#elif defined(__GNUC__)
  return __atomic_fetch_add(x, y, __ATOMIC_SEQ_CST);
#elif defined(_MSC_VER)
  return InterlockedExchangeAdd(x, y);
#elif defined(__clang__)
  return __c11_atomic_fetch_add(x, y)
#else
#error "No atomic_fetch_add implementation."
#endif
}


template<typename Integer64>
__host__ __device__
typename enable_if<
  sizeof(Integer64) == 8,
  Integer64
>::type
atomic_fetch_add(Integer64 *x, Integer64 y)
{
#if defined(__CUDA_ARCH__)
  return atomicAdd(x, y);
#elif defined(__GNUC__)
  return __atomic_fetch_add(x, y, __ATOMIC_SEQ_CST);
#elif defined(_MSC_VER)
  return InterlockedExchangeAdd64(x, y);
#elif defined(__clang__)
  return __c11_atomic_fetch_add(x, y)
#else
#error "No atomic_fetch_add implementation."
#endif
}


template<typename Integer32>
__host__ __device__
typename enable_if<
  sizeof(Integer32) == 4,
  Integer32
>::type
atomic_fetch_sub(Integer32 *x, Integer32 y)
{
#if defined(__CUDA_ARCH__)
  return atomicSub(x, y);
#elif defined(__GNUC__)
  return __atomic_fetch_sub(x, y, __ATOMIC_SEQ_CST);
#elif defined(_MSC_VER)
  return InterlockedExchangeAdd(x, -y);
#elif defined(__clang__)
  return __c11_atomic_fetch_sub(x, y)
#else
#error "No atomic_fetch_sub implementation."
#endif
}


template<typename Integer64>
__host__ __device__
typename enable_if<
  sizeof(Integer64) == 8,
  Integer64
>::type
atomic_fetch_sub(Integer64 *x, Integer64 y)
{
#if defined(__CUDA_ARCH__)
  return atomicSub(x, y);
#elif defined(__GNUC__)
  return __atomic_fetch_sub(x, y, __ATOMIC_SEQ_CST);
#elif defined(_MSC_VER)
  return InterlockedExchangeAdd64(x, -y);
#elif defined(__clang__)
  return __c11_atomic_fetch_sub(x, y)
#else
#error "No atomic_fetch_sub implementation."
#endif
}


template<typename Integer>
class atomic_base
{
  private:
    typedef Integer int_type;

  public:
    __host__ __device__
    atomic_base()
    {}

    __host__ __device__
    atomic_base(int_type i)
      : x(i)
    {}

    __host__ __device__
    operator int_type() const
    {
      return load();
    }

    __host__ __device__
    operator int_type() const volatile
    {
      return load();
    }

    __host__ __device__
    int_type operator=(int_type val)
    {
      store(val);
      return val;
    }

    __host__ __device__
    int_type operator++(int)
    {
      return fetch_add(1);
    }

    __host__ __device__
    int_type operator++(int) volatile
    {
      return fetch_add(1);
    }

    __host__ __device__
    int_type operator--(int)
    {
      return fetch_sub(1);
    }

    __host__ __device__
    int_type operator--(int) volatile
    {
      return fetch_sub(1);
    }

    __host__ __device__
    int_type operator++()
    {
      // return atomic_add_fetch(&x, 1);
      return operator++(0) + int_type(1);
    }

    __host__ __device__
    int_type operator++() volatile
    {
      // return atomic_add_fetch(&x, 1);
      return operator++(0) + int_type(1);
    }

    __host__ __device__
    int_type operator--()
    {
      // return atomic_sub_fetch(&x, 1);
      return operator--(0) - int_type(1);
    }

    __host__ __device__
    int_type operator--() volatile
    {
      // return atomic_sub_fetch(&x, 1);
      return operator--(0) - int_type(1);
    }

    __host__ __device__
    int_type fetch_add(int_type i)
    {
      return detail::atomic_fetch_add(&x, i);
    }

    __host__ __device__
    int_type fetch_add(int_type i) volatile
    {
      return detail::atomic_fetch_add(const_cast<int_type*>(&x), i);
    }

    __host__ __device__
    int_type fetch_sub(int_type i)
    {
      return detail::atomic_fetch_sub(&x, i);
    }

    __host__ __device__
    int_type fetch_sub(int_type i) volatile
    {
      return detail::atomic_fetch_sub(const_cast<int_type*>(&x), i);
    }

    __host__ __device__
    void store(int_type i)
    {
      detail::atomic_store_n(&x, i);
    }

    __host__ __device__
    void store(int_type i) volatile
    {
      detail::atomic_store_n(const_cast<int_type*>(&x), i);
    }

    __host__ __device__
    int_type load() const
    {
      return detail::atomic_load_n(&x);
    }

    __host__ __device__
    int_type load() const volatile
    {
      return detail::atomic_load_n(&x);
    }

  protected:
    atomic_base(const atomic_base &);

    atomic_base &operator=(const atomic_base&);

    atomic_base &operator=(const atomic_base&) volatile;

  private:
    int_type x;
};


} // end detail


typedef detail::atomic_base<int>                atomic_int;

typedef detail::atomic_base<unsigned int>       atomic_uint;

typedef detail::atomic_base<long>               atomic_long;

typedef detail::atomic_base<unsigned long>      atomic_ulong;

typedef detail::atomic_base<long long>          atomic_llong;

typedef detail::atomic_base<unsigned long long> atomic_ullong;

typedef detail::atomic_base<size_t>             atomic_size_t;


template<typename T> class atomic;


template<>
class atomic<int> : public atomic_int
{
  private:
    typedef atomic_int super_t;

    atomic(const atomic &);

    atomic &operator=(const atomic&);

    atomic &operator=(const atomic&) volatile;

  public:
    __host__ __device__
    atomic() : super_t() {}

    __host__ __device__
    atomic(int x) : super_t(x) {}

    using super_t::operator int;

    using super_t::operator=;
};


template<>
class atomic<unsigned int> : public atomic_uint
{
  private:
    typedef atomic_uint super_t;

    atomic(const atomic &);

    atomic &operator=(const atomic&);

    atomic &operator=(const atomic&) volatile;

  public:
    __host__ __device__
    atomic() : super_t() {}

    __host__ __device__
    atomic(unsigned int x) : super_t(x) {}

    using super_t::operator unsigned int;

    using super_t::operator=;
};


template<>
class atomic<long> : public atomic_long
{
  private:
    typedef atomic_long super_t;

    atomic(const atomic &);

    atomic &operator=(const atomic&);

    atomic &operator=(const atomic&) volatile;

  public:
    __host__ __device__
    atomic() : super_t() {}

    __host__ __device__
    atomic(long x) : super_t(x) {}

    using super_t::operator long;

    using super_t::operator=;
};


template<>
class atomic<unsigned long> : public atomic_ulong
{
  private:
    typedef atomic_ulong super_t;

    atomic(const atomic &);

    atomic &operator=(const atomic&);

    atomic &operator=(const atomic&) volatile;

  public:
    __host__ __device__
    atomic() : super_t() {}

    __host__ __device__
    atomic(unsigned long x) : super_t(x) {}

    using super_t::operator unsigned long;

    using super_t::operator=;
};


template<>
class atomic<long long> : public atomic_llong
{
  private:
    typedef atomic_llong super_t;

    atomic(const atomic &);

    atomic &operator=(const atomic&);

    atomic &operator=(const atomic&) volatile;

  public:
    __host__ __device__
    atomic() : super_t() {}

    __host__ __device__
    atomic(long long x) : super_t(x) {}

    using super_t::operator long long;

    using super_t::operator=;
};


template<>
class atomic<unsigned long long> : public atomic_ullong
{
  private:
    typedef atomic_ullong super_t;

    atomic(const atomic &);

    atomic &operator=(const atomic&);

    atomic &operator=(const atomic&) volatile;

  public:
    __host__ __device__
    atomic() : super_t() {}

    __host__ __device__
    atomic(unsigned long long x) : super_t(x) {}

    using super_t::operator unsigned long long;

    using super_t::operator=;
};


} // end nuke


#ifdef NUKE_UNDEF_HOST
#undef __host__
#undef NUKE_UNDEF_HOST
#endif


#ifdef NUKE_UNDEF_DEVICE
#undef __device__
#undef NUKE_UNDEF_DEVICE
#endif

