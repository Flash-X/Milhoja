#ifndef GRID_VECTOR_H__
#define GRID_VECTOR_H__

#include <vector>

namespace orchestration {

template <class T, class Allocator=std::allocator<T>> class Vector : public std::vector<T, Allocator>
{
public:
    using std::vector<T>::vector;

    template <class W>
    Vector<T> (const Vector<W>& vec){
      Vector<T> result;
      result.reserve(vec.size());
      std::transform(vec.begin(), vec.end(), std::back_inserter(result), [](const W& x) -> T { return x; });
      *this= result;
    }
};

//Global overload of + operator. Not sure if this is better than making it a member function.
template <class T>
Vector<T> operator+(const Vector<T>& a, const Vector<T>& b)
{
    //TODO how to do assert statements
    //assert(a.size() == b.size());

    Vector<T> result;
    result.reserve(a.size());
    std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(result), std::plus<T>());
    return result;
}



}

#endif
