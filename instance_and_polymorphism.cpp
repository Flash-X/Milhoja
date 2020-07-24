#include <iostream>
using namespace std;
#define GRID_AMREX

// In some Grid header
#ifdef GRID_AMREX
class B;
typedef B derivedA;
#else
class A;
typedef A derivedA;
#endif 

// A.h
class A {
    public:
    static A& instance() ;
    
    virtual void foo(int vin) {
        std::cout << "A foo: " << vin <<std::endl;
    }
    
    int val;

    protected:
        A () { val = 0; }
};

//B.h
class B : public A {
    public:
    void foo(int vin) override {
        std::cout << "B foo: " << vin*2 <<std::endl;
    }
    
    protected:
        B () { val = 23; }
        friend A& A::instance();
};

//A.cpp
A& A::instance() {
    static derivedA singleton;
    return singleton;
}

int main()
{
    //A aObj; // fails
    A& aObj = A::instance(); //ok
    
    aObj.foo(4); // ifdef GRID_AMREX: "B foo: 8", else: "A foo: 4"

    return 0;
}
