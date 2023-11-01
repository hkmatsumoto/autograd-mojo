import math


@register_passable
struct Node:
    # HACK: Since we don't have ADTs or inheritance yet, Node represents input
    # nodes, intermediate nodes, and constant nodes combined.

    var grad: Float32

    # Intermediate variable
    var operation: Operation
    var op1: Pointer[Int]
    var op2: Pointer[Int]

    # Input and constant
    var value: Float32

    fn __init__(value: Float32) -> Node:
        let grad = 0.0

        let operation = Operation.nop
        let op1 = Pointer[Int].get_null()
        let op2 = Pointer[Int].get_null()

        return Node {grad: grad, operation: operation, op1: op1, op2: op2, value: value}

    fn __init__(operation: Operation, op1: Pointer[Int], op2: Pointer[Int]) -> Node:
        let grad = 1.0
        let value = 0.0

        return Node {grad: grad, operation: operation, op1: op1, op2: op2, value: value}

    fn forward(self) raises -> Float32:
        if self.operation == Operation.nop:
            return self.value

        let op1 = self.op1.bitcast[Node]().load().forward()
        let op2 = self.op2.bitcast[Node]().load().forward()

        if self.operation == Operation.add:
            return op1 + op2
        elif self.operation == Operation.sub:
            return op1 - op2
        elif self.operation == Operation.mul:
            return op1 * op2
        elif self.operation == Operation.div:
            return op1 / op2
        elif self.operation == Operation.relu:
            return op1 if op1 > 0.0 else 0.0
        elif self.operation == Operation.sigmoid:
            return 1.0 / (1.0 + math.exp(-op1))
        else:
            raise Error("Invalid operation")

    fn backward(self) raises -> None:
        var op1 = self.op1.bitcast[Node]().load()
        var op2 = self.op2.bitcast[Node]().load()

        if self.operation == Operation.add:
            op1.grad = self.grad
            op2.grad = self.grad
        elif self.operation == Operation.sub:
            op1.grad = self.grad
            op2.grad = -self.grad
        elif self.operation == Operation.mul:
            op1.grad = self.grad * op2.forward()
            op2.grad = self.grad * op1.forward()
        elif self.operation == Operation.div:
            op1.grad = self.grad / op2.forward()
            op2.grad = -self.grad * op1.forward() / (op2.forward() * op2.forward())
        elif self.operation == Operation.relu:
            op1.grad = self.grad if op1.forward() > 0.0 else 0.0
        elif self.operation == Operation.sigmoid:
            op1.grad = self.grad * self.forward() * (1.0 - self.forward())
        else:
            raise Error("Invalid operation")

        self.op1.bitcast[Node]().store(op1)
        self.op2.bitcast[Node]().store(op2)

        if op1.operation != Operation.nop:
            op1.backward()
        if op2.operation != Operation.nop:
            op2.backward()


@value
@register_passable
struct Operation:
    var opcode: UInt8

    alias nop = Operation(0)
    alias add = Operation(1)
    alias sub = Operation(2)
    alias mul = Operation(3)
    alias div = Operation(4)

    alias relu = Operation(10)
    alias sigmoid = Operation(11)

    fn __eq__(self, other: Operation) -> Bool:
        return self.opcode == other.opcode

    fn __ne__(self, other: Operation) -> Bool:
        return self.opcode != other.opcode


def main():
    let x = Node(1.0)
    let y = Node(2.0)
    let z = Node(4.0)

    let op1 = Pointer[Node].alloc(1)
    op1.store(x)

    let op2 = Pointer[Node].alloc(1)
    op2.store(y)

    let op3 = Pointer[Node].alloc(1)
    op3.store(z)

    let add = Node(Operation.add, op1.bitcast[Int](), op2.bitcast[Int]())
    let op4 = Pointer[Node].alloc(1)
    op4.store(add)

    let mul = Node(Operation.mul, op3.bitcast[Int](), op4.bitcast[Int]())
    let op5 = Pointer[Node].alloc(1)
    op5.store(mul)

    let dummy = Node(0.0)
    let op_dummy = Pointer[Node].alloc(1)
    op_dummy.store(dummy)

    let relu = Node(Operation.relu, op5.bitcast[Int](), op_dummy.bitcast[Int]())
    print("Forward:", relu.forward())

    relu.backward()
    print("Gradient of x:", op1.load().grad)
    print("Gradient of y:", op2.load().grad)
    print("Gradient of z:", op3.load().grad)
