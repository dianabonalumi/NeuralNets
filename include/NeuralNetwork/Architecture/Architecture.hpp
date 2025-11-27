#ifndef ARCHITECTURE_HPP
#define ARCHITECTURE_HPP

#include "../Matrix.hpp"

template<typename T>
class Architecture {
public:
    virtual void Train(const std::vector<T>) = 0;
    virtual const std::vector<T> Eval(const std::vector<T>) = 0;
};

#endif