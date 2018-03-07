import numpy as np

# assume we have a belief of the position the robot is at, represented as an array
# prior = [.1, .1, .1, .1, .1, .1, .1, .1, .1]
# where prior[0] = 0.1 means the probability that the robot is at pos 0 is 10%
# say we sensed that the robot moved toward right by n units, but the sensor is noisy
# we use a smaller array to represent that
# kernel = [.1, .8, .1]
# where kernel[1] = .8 means the probability that the robot moved as the sensor reported
# is .8 whereas kernel[2] = .1 means the probability that the robot moved one unit furhter to
# the right than reported is .1.
# after incorporating the movement to our prior we have adjusted belief and it can be computed
# using convolution as the following method

# the method is tailored for our problem. The middle element in kernel is the probability that
# the robot moved as reported. The other elements are the probabilities that the robot moved more to
# the right or left by some units. The exact direction and number of units are determined by the relative positions
# the elements are to the middle elements. If one element is 2 positions away from the middle on the left, it represents
# the probability that the robot moved toward the right by n - 2 units. So the length of kernel array is always an odd number

# convolution computing can have other forms

def predict_move_convolution(pdf, offset, kernel):
  N = len(pdf)
  kN = len(kernel)
  width = int((kN - 1) / 2)

  prior = np.zeros(N)
  for i in range(N):
    for k in range (kN):
      index = (i + (width-k) - offset) % N
      prior[i] += pdf[index] * kernel[k]
  return prior

# the method above does compute the result correctly. But it doesn't compute it the way how kernel
# slides through the prior array. Let's try to compute convolution as the kernel slides through the prior

def predict_move_convolution_slide(pdf, offset, kernel):
  N = len(pdf)
  kN = len(kernel)
  width = int((kN - 1) / 2)
  # indexes represent which elements in prior that the left most and right most elements in kernel are at
  right_index = 0
  left_index = right_index - kN + 1
  sliding_steps = N + kN - 1

  prior = np.zeros(N)
  for i in range(sliding_steps):
    # use two indexes each to track the overlapped part of the two arrays
    kernel_left = 0
    kernel_right = kN - 1
    left = left_index
    right = right_index
    if left_index < 0:
      left = 0
      kernel_left = kernel_right - (right - left)
    if right_index > N - 1:
      right = N - 1
      kernel_right = kernel_left + (right - left)
    product = kernel[kernel_left:kernel_right + 1] * pdf[left:right+1]
    index_k = kernel_left
    index_pdf = left
    for k in range(len(product)):
      index_p = (index_pdf + offset + (width - index_k)) % N
      prior[index_p] += product[k]

    left_index += 1
    right_index += 1

  return prior

pdf = np.array([.05, .05, .05, .05, .55, .05, .05, .05, .05, .05])
offset = 1
kernel = np.array([.1, .8, .1])

# check that they produce the same result

print("conv 1 results in: ", predict_move_convolution(pdf, offset, kernel))
print()
print("conv 2 results in: ", predict_move_convolution_slide(pdf, offset, kernel))
