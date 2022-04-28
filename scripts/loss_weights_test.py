"""
    for testing values for the weights of the separable flow loss function
"""

if __name__ == "__main__":

    # inital flow regression images
    n_img_regression = 3
    
    # flow refinement step images
    n_img_refinement = 5

    # total number of flow images for loss computation
    n_predictions = n_img_regression + n_img_refinement
    
    # weight decay
    gamma = 0.8

    # weights for the flow regression images
    weights = [0.1, 0.3, 0.5]
    
    # the base weight value is the weight for weights[2] - gamma**n_img_refinement
    base = weights[2] - gamma ** (n_predictions - 3)
    print(f"base weight value: {base}")

    # rest of the weights: weights[2] - gamma**n_img_refinement + gamma**(n_img_refinement-i-i)
    for i in range(n_predictions - 3):
        weights.append( base + gamma**(n_predictions - i - 4) )

    # weights for n_img_refinement = 4:
    #   [0.1, 0.3, 0.5, 0.6024, 0.7304, 0.8904, 1.0903999999999998]
    # weigths for n_img_refinement = 5:
    #   [0.1, 0.3, 0.5, 0.58192, 0.68432, 0.81232, 0.97232, 1.17232]
    print(f"final weights: {weights}")