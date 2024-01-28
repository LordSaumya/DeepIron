/// A set of structs and functions that define a layer in a neural network.
///
/// # Example
///
/// ```
/// let layer = layer::new(Series::new("weights", vec![1.0, 2.0, 3.0]), Series::new("biases", vec![1.0, 2.0, 3.0]));
/// ```
///
pub mod layer {
    use crate::model::activation_functions::{ActivationFunction, ActivationFunctionType};
    use crate::model::loss_functions::{LossFunction, LossFunctionType};
    use polars::prelude::*;
    use rand::{thread_rng, Rng};
    /// A trait that defines a set of functions for a layer in a neural network.
    pub trait Layer {
        /// Performs a forward pass on the layer using the given activation function, calculating the outputs of the layer.
        ///
        /// # Example
        ///
        /// ```
        /// let layer = LinearLayer::new(Series::new("weights", vec![1.0, 2.0, 3.0]), Series::new("biases", vec![1.0, 2.0, 3.0]));
        ///
        /// let output = layer.forward(Series::new("inputs", vec![1.0, 2.0, 3.0]), ActivationFunctionType::ReLU);
        ///
        /// ```
        ///
        /// # Arguments
        ///
        /// * `inputs` - A series of inputs to the layer.
        ///
        /// * `activation_function` - The activation function to use for the layer.
        ///
        /// # Returns
        ///
        /// A series of outputs from the layer.
        fn forward(&self, inputs: Series, activation_function: ActivationFunctionType) -> Series;

        /// Performs a backward pass on the layer using the given loss function, calculating the gradients of the weights and biases for each layer.
        ///
        /// # Example
        ///
        /// ```
        /// let layer = LinearLayer::new(Series::new("weights", vec![1.0, 2.0, 3.0]), Series::new("biases", vec![1.0, 2.0, 3.0]));
        ///
        /// let gradients = layer.backward(Series::new("inputs", vec![1.0, 2.0, 3.0]), Series::new("outputs", vec![1.0, 2.0, 3.0]), LossFunctionType::BinaryCrossEntropy, Series::new("upstream_gradient", vec![1.0, 2.0, 3.0]));
        ///
        /// ```
        ///
        /// # Arguments
        ///
        /// * `inputs` - A series of inputs to the layer.
        ///
        /// * `outputs` - A series of outputs from the layer.
        ///
        /// * `loss_function` - The loss function to use for the layer.
        ///
        /// * `activation_function` - The activation function to use for the layer.
        ///
        /// * `upstream_gradient` - The gradients from the subsequent layer.
        ///
        /// # Returns
        ///
        /// A tuple of three series of gradients, corresponding to the gradients of the weights, biases, and inputs, respectively.
        fn backward(
            &self,
            inputs: Series,
            outputs: Series,
            loss_function: LossFunctionType,
            activation_function: ActivationFunctionType,
            upstream_gradient: Series,
        ) -> (Series, Series, Series);
    }

    /// A struct that defines a linear (fully-connected) layer in a neural network.
    ///
    /// # Example
    ///
    /// ```
    /// let linear_layer = LinearLayer::new(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
    /// ```
    #[derive(Clone)]
    pub struct LinearLayer {
        pub weights: Series,
        pub biases: Series,
    }

    impl LinearLayer {
        /// Create a new layer with the given weights and biases.
        ///
        /// # Example
        ///
        /// ```
        /// let layer = LinearLayer::new(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        /// ```
        ///
        /// # Arguments
        ///
        /// * `weights` - A series of weights for the layer.
        ///
        /// * `biases` - A series of biases for the layer.
        ///
        /// # Returns
        ///
        /// A new layer with the given weights and biases.
        pub fn new(weights: Series, biases: Series) -> LinearLayer {
            LinearLayer { weights, biases }
        }

        /// Create a new linear layer with all weights and biases set to zero.
        ///
        /// # Example
        ///
        /// ```
        /// let layer = LinearLayer::zeroes(3);
        /// ```
        ///
        /// # Arguments
        ///
        /// * `width` - The width of the layer.
        ///
        /// # Returns
        ///
        /// A new linear layer of the given width with all weights and biases set to zero.
        pub fn zeroes(width: usize) -> LinearLayer {
            LinearLayer {
                weights: Series::new("weights", vec![0.0; width]),
                biases: Series::new("biases", vec![0.0; width]),
            }
        }

        /// Create a new linear layer with all weights and biases set to random values uniformly distributed within the given range.
        ///
        /// # Example
        ///
        /// ```
        /// let layer = LinearLayer::new_random(3, [-1.0, 1.0]);
        /// ```
        ///
        /// # Arguments
        ///
        /// * `width` - The width of the layer.
        ///
        /// * `range` - The range of values to use for the random weights and biases.
        ///
        /// # Returns
        ///
        /// A new linear layer of the given width with all weights and biases set to random values uniformly distributed within the given range (inclusive).
        pub fn new_random(width: usize, range: [f64; 2]) -> LinearLayer {
            let mut init_weights: Vec<f64> = Vec::with_capacity(width);
            let mut init_biases: Vec<f64> = Vec::with_capacity(width);
            let low: f64 = if range[0] < range[1] {
                range[0]
            } else {
                range[1]
            };
            let high: f64 = if range[0] < range[1] {
                range[1]
            } else {
                range[0]
            };

            let mut rng: rand::prelude::ThreadRng = thread_rng();

            for _ in 0..width {
                init_weights.push(rng.gen_range(low..=high));
                init_biases.push(rng.gen_range(low..=high));
            }

            LinearLayer {
                weights: Series::new("weights", init_weights),
                biases: Series::new("biases", init_biases),
            }
        }
    }

    impl Layer for LinearLayer {
        fn forward(&self, inputs: Series, activation_function: ActivationFunctionType) -> Series {
            let dot_product: f64 = (&inputs * &self.weights).sum().unwrap();
            let dot_prod_series = Series::new("dot_product", vec![dot_product; inputs.len()]);
            activation_function.activate(&(dot_prod_series + self.biases.clone()))
        }

        fn backward(
            &self,
            inputs: Series,                              // X
            outputs: Series,                             // Z
            loss_function: LossFunctionType,             // L
            activation_function: ActivationFunctionType, // A
            upstream_gradients: Series,                  // dL/dZ
        ) -> (Series, Series, Series) {
            let activation_gradients: Series = activation_function.gradient(&outputs); // dA/dZ
            let weighted_gradients: Series = &activation_gradients * &upstream_gradients; // dL/dZ * dA/dZ
            let weight_gradients: Series = &inputs * &weighted_gradients; // dW/dX = X * dL/dZ * dA/dZ
            let bias_gradients: Series = weighted_gradients; // dB/dX = dL/dZ * dA/dZ
            let input_gradients: Series = loss_function.gradient(&inputs.into_frame(), 
                &outputs, &upstream_gradients); // dL/dX = dL/dZ * dA/dZ * dZ/dX
            (
                weight_gradients, // dW/dX
                bias_gradients,   // dB/dZ
                input_gradients,  // dL/dX
            )
        }
    }
}
