/**
 * \file    Milhoja_RuntimeElement.h
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

#ifndef MILHOJA_RUNTIME_ELEMENT_H__
#define MILHOJA_RUNTIME_ELEMENT_H__

#include <map>
#include <string>
#include <memory>

#include "Milhoja_DataItem.h"

namespace milhoja {

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
    virtual void         closeQueue(const RuntimeElement* publisher) = 0;

    virtual std::string  attachDataReceiver(RuntimeElement* receiver);
    virtual std::string  detachDataReceiver(void);

    virtual std::string  setReceiverPrototype(const DataItem* prototype);

protected:
    RuntimeElement(void);
    virtual ~RuntimeElement(void);

    std::string    attachDataPublisher(const RuntimeElement* publisher);
    std::string    detachDataPublisher(const RuntimeElement* publisher);

    RuntimeElement*   threadReceiver_; //!< RuntimeElement to notify when threads terminate
    RuntimeElement*   dataReceiver_;   /*!< RuntimeElement to pass data items
                                            to once this team's action has
                                            already been applied to the
                                            items. */
    const DataItem*  receiverPrototype_;

    std::map<const RuntimeElement*,bool>   calledCloseQueue_;  /*!< The keys in this map serve as a list
                                                                    of data publishers attached to the object.
                                                                    Values indicate which publishers have
                                                                    called the object's closeQueue member
                                                                    function in the current runtime
                                                                    execution cycle.  Derived classes must
                                                                    determine if this variable needs to
                                                                    be managed in a thread-safe way and 
                                                                    to do so when and where necessary.*/
};

}

#endif

