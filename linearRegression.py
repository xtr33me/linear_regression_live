from numpy import *

def compute_error_for_line_given_points(b, m, points):
    #init at 0
    totalError = 0
    #for every points
    for i in range(0, len(points)):
        #get the x val
        x = points[i,0]
        #get the y val
        y = points[i,1]
        #compute difference, square it and add it to the total
        totalError += (y - (m * x + b)) **2
    #get the average
    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    #starting value for b and m
    b = starting_b
    m = starting_m

    #gradient descent
    for i in range(num_iterations):
        #update b and m with more accurate b and m
        b, m = step_gradient(b, m, array(points), learning_rate)
    
    return [b,m]

def step_gradient(b_current, m_current, points, learning_rate):
    #starting points for gradient
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        #direction with respect to b and m
        #computing partial derivatives with respect to b and m of our error function
        #This will provide us the directions to go for both b and m
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y- ((m_current * x) + b_current))
    
    #update our b and m values using the partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]



def run():
    #Step 1 - collect our data
    points = genfromtxt('data.csv', delimiter=',')

    #step 2 - define hyperparameters (tuning knobs)
    #how fast should the model converge
    learning_rate = 0.0001
    #y=mx+b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    #step 3 - train the model
    print 'Starting gradient descent at b = {0}, m={1}, error={2}'.format( initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points) )
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print 'Iterations: {0} - Ending point at b = {1}, m={2}, error={3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))



if __name__ == '__main__':
    run()