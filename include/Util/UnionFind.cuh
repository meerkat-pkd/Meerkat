#ifndef UNION_FIND_CUH_
#define UNION_FIND_CUH_

#include <cstdint>
#include <tuple>
#include <type_traits>

template <typename T, typename... Args> struct AnyOf;

template <typename T, typename FirstType, typename... OtherTypes>
struct AnyOf<T, FirstType, OtherTypes...> {
  static constexpr bool value =
      std::is_same<T, FirstType>::value || AnyOf<T, OtherTypes...>::value;
  using value_type = bool;
  using type = std::integral_constant<bool, value>;
};

template <typename T, typename FirstType> struct AnyOf<T, FirstType> {
  static constexpr bool value = std::is_same<T, FirstType>::value;
  using value_type = bool;
  using type = std::integral_constant<bool, value>;
};

namespace find_op {
struct FindNaive {
  static __device__ uint32_t Find(uint32_t TheVertex, uint32_t *Parents) {
    uint32_t V = TheVertex;

    while (V != Parents[V])
      V = Parents[V];
    return V;
  }
};

struct FindCompress {
  static __device__ uint32_t Find(uint32_t i, uint32_t *Parents) {
    uint32_t j = i;
    if (Parents[j] == j)
      return j;

    do {
      j = Parents[j];
    } while (Parents[j] != j);

    uint32_t tmp;
    while ((tmp = Parents[i]) > j) {
      Parents[i] = j;
      i = tmp;
    }

    return j;
  }
};

struct AtomicSplit {
  static __device__ uint32_t Find(uint32_t i, uint32_t *Parents) {
    while (1) {
      uint32_t v = Parents[i];
      uint32_t w = Parents[v];
      if (v == w) {
        return v;
      } else {
        atomicCAS(&Parents[i], v, w);
        // parent[i] = w;
        i = v;
      }
    }
  }
};

struct AtomicHalve {
  static __device__ uint32_t Find(uint32_t i, uint32_t *Parents) {
    while (1) {
      uint32_t v = Parents[i];
      uint32_t w = Parents[v];
      if (v == w) {
        return v;
      } else {
        atomicCAS(&Parents[i], v, w);
        // parent[i] = w;
        i = Parents[i];
      }
    }
  }
};
} // namespace find_op

namespace splice_op {
struct AtomicSplitOne {
  static __device__ uint32_t Splice(uint32_t TheVertex, uint32_t _,
                                    uint32_t *Parents) {
    uint32_t V = Parents[TheVertex];
    uint32_t W = Parents[V];
    if (V != W)
      atomicCAS(&Parents[TheVertex], V, W);
    return V;
  }
};

struct AtomicHalveOne {
  static __device__ uint32_t Splice(uint32_t TheVertex, uint32_t _,
                                    uint32_t *Parents) {
    uint32_t V = Parents[TheVertex];
    uint32_t W = Parents[V];
    if (V != W)
      atomicCAS(&Parents[TheVertex], V, W);
    return W;
  }
};

struct SpliceAtomic {
  static __device__ uint32_t Splice(uint32_t TheVertex, uint32_t OtherVertex,
                                    uint32_t *Parents) {
    uint32_t Parent = Parents[TheVertex];
    atomicMin(&Parents[TheVertex], Parents[OtherVertex]);
    return Parent;
  }
};
} // namespace splice_op

namespace union_op {
template <typename Find,
          typename std::enable_if<
              AnyOf<Find, find_op::FindNaive, find_op::AtomicSplit,
                    find_op::AtomicHalve, find_op::FindCompress>::value>::type
              * = nullptr>
struct UnionAsync {
  using FindStrategy = Find;
  static __device__ bool Union(uint32_t src, uint32_t dst, uint32_t *parents) {
    while (1) {
      uint32_t u = Find::Find(src, parents);
      uint32_t v = Find::Find(dst, parents);
      if (u == v)
        break;
      if (v > u) {
        uint32_t temp;
        temp = u;
        u = v;
        v = temp;
      }
      if (u == atomicCAS(&parents[u], u, v)) {
        //   if(parents(u) == u && u == atomicCAS(&parents(u),u,v)) {
        return true;
      } else {
      }
    }
    return false;
  }
};

template <typename Find,
          typename std::enable_if<
              AnyOf<Find, find_op::FindNaive, find_op::AtomicSplit,
                    find_op::AtomicHalve, find_op::FindCompress>::value>::type
              * = nullptr>
struct UnionAsync1 {
  using FindStrategy = Find;
  static __device__ bool Union(uint32_t src, uint32_t dst, uint32_t *parents, uint32_t *Count) {
    while (1) {
      uint32_t u = Find::Find(src, parents);
      uint32_t v = Find::Find(dst, parents);
      if (u == v)
        break;
      if (v > u) {
        uint32_t temp;
        temp = u;
        u = v;
        v = temp;
      }
      if (u == atomicCAS(&parents[u], u, v)) {
        //   if(parents(u) == u && u == atomicCAS(&parents(u),u,v)) {
        atomicAdd(Count, 1);
        return true;
      } else {
      }
    }
    return false;
  }
};

template <typename Find,
          typename std::enable_if<
              AnyOf<Find, find_op::FindNaive, find_op::AtomicSplit,
                    find_op::AtomicHalve, find_op::FindCompress>::value>::type
              * = nullptr>
struct UnionEarly {
  using FindStrategy = Find;
  static __device__ bool Union(uint32_t U, uint32_t V, uint32_t *Parents) {
    uint32_t ParentU = U;
    uint32_t ParentV = V;

    while (ParentU != ParentV) {
      if (ParentV > ParentU) {
        uint32_t T = ParentV;
        ParentV = ParentU;
        ParentU = T;
      }

      if (ParentU == Parents[ParentU] &&
          atomicCAS(&Parents[U], ParentU, ParentV) == ParentU)
        break;

      uint32_t Z = Parents[ParentU];
      uint32_t W = Parents[Z];
      atomicCAS(&Parents[ParentU], Z, W);
      ParentU = W;
    }

    if (!std::is_same<Find, find_op::FindNaive>::value)
      ParentU = Find::Find(U, Parents);

    if (!std::is_same<Find, find_op::FindNaive>::value)
      ParentV = Find::Find(V, Parents);

    return true;
  }
};

template <typename Compress, typename SpliceAtomic,
          typename std::enable_if<
              AnyOf<Compress, find_op::FindNaive, find_op::AtomicSplit,
                    find_op::AtomicHalve>::value>::type * = nullptr,
          typename std::enable_if<AnyOf<
              SpliceAtomic, splice_op::SpliceAtomic, splice_op::AtomicSplitOne,
              splice_op::AtomicHalveOne>::value>::type * = nullptr>
struct UnionRemCAS {
  using FindStrategy = Compress;
  static __device__ bool Union(uint32_t U, uint32_t V, uint32_t *Parents) {
    uint32_t R_U = U;
    uint32_t R_V = V;

    while (Parents[R_U] != Parents[R_V]) {
      if (R_V > R_U) {
        uint32_t T = R_V;
        R_V = R_U;
        R_U = T;
      }

      if (R_U == Parents[R_U] &&
          atomicCAS(&Parents[R_U], R_U, Parents[R_V]) == R_U) {
        if (!std::is_same<Compress, find_op::FindNaive>::value) {
          Compress::Find(U, Parents);
          Compress::Find(V, Parents);
        }
        return true;
      } else {
        R_U = SpliceAtomic::Splice(R_U, R_V, Parents);
      }
    }

    return false;
  }
};
} // namespace union_op

#endif // UNION_FIND_CUH_