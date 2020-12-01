/**
 * \file    RuntimeElement.h
 *
 * \brief An abstract base class for all elements that can be used to compose a
 * thread team configuration.
 *
 * This abstract base class defines the interface for the thread and data
 * publisher/subscriber facilities of the runtime's thread team configuration.
 * Common functionality is implemented.
 *
 * The implementations of the data/thread publisher/subscriber design aspects
 * are one-directional versions of the Observer design pattern in the Gang of
 * Four design patterns book (Pp. 293).
 */

#ifndef RUNTIME_ELEMENT_H__
#define RUNTIME_ELEMENT_H__

#include <string>
#include <memory>

#include "DataItem.h"

namespace orchestration {

class RuntimeElement {
public:
    RuntimeElement(RuntimeElement&)                  = delete;
    RuntimeElement(const RuntimeElement&)            = delete;
    RuntimeElement(RuntimeElement&&)                 = delete;
    RuntimeElement& operator=(RuntimeElement&)       = delete;
    RuntimeElement& operator=(const RuntimeElement&) = delete;
    RuntimeElement& operator=(RuntimeElement&&)      = delete;

    // Thread Publisher/Subscriber interface
    virtual void increaseThreadCount(const unsigned int nThreads) = 0;

    virtual std::string  attachThreadReceiver(RuntimeElement* receiver);
    virtual std::string  detachThreadReceiver(void);

    // Data Publisher/Subscriber interface
    virtual void         enqueue(std::shared_ptr<DataItem>&& dataItem) = 0;
    virtual void         closeQueue(void) = 0;

    virtual std::string  attachDataReceiver(RuntimeElement* receiver);
    virtual std::string  detachDataReceiver(void);

protected:
    RuntimeElement(void);
    virtual ~RuntimeElement(void);

    RuntimeElement*   threadReceiver_; //!< RuntimeElement to notify when threads terminate
    RuntimeElement*   dataReceiver_;   /*!< RuntimeElement to pass data items
                                            to once this team's action has
                                            already been applied to the
                                            items. */
};

}

#endif

