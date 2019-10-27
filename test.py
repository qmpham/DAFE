import tensorflow as tf
tf.random.set_seed(0)
v = tf.Variable(3.0) #tf.Variable(tf.random.normal((2,)))
with tf.GradientTape(persistent=True) as tape:
    y = tf.nn.relu(v) #tf.sin(v[0] + tf.cos(v[1]))
    grad = tape.gradient(y, v)
    #hessian = tape.gradient(grad, v)
    #hessian0 = tape.gradient(grad[0], v)
    #hessian1 = tape.gradient(grad[1], v)
"""
print("hessian:", hessian)
print("hessian0:", hessian0)
print("hessian1:", hessian1)
print("hessian0 + hessian1:", hessian0 + hessian1)
"""
print("grad: ", grad)
