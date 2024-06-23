//! A set of structs and functions that define a layer in a neural network.

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
        /// let layer: LinearLayer = LinearLayer::new(Series::new("weights", vec![1.0, 2.0, 3.0]), Series::new("biases", vec![1.0, 2.0, 3.0]));
        /// 
        /// let (updated_weights, updated_biases) = layer.backward(Series::new("inputs", vec![1.0, 2.0, 3.0]), Series::new("grad_outputs", vec![1.0, 2.0, 3.0], 0.01);
        /// ```
        ///
        /// # Arguments
        /// 
        /// * `inputs` - A series of inputs to the layer.
        /// 
        /// * `grad_outputs` - A series of gradients of the outputs of the layer.
        /// 
        /// * `lr` - The learning rate to use for the update.
        /// 
        /// # Returns
        /// 
        /// A tuple of the updated weights and biases for the layer.
        /// 
        fn backward(&self, inputs: Series, grad_outputs: Series, lr: f64) -> (Series, Series);
    }

    /// A struct that defines a linear (fully-connected) layer in a neural network.
    ///
    /// # Example
    ///
    /// ```
    /// let linear_layer = LinearLayer::new(Series::new("weights", vec![1.0, 2.0, 3.0]), Series::new("biases", vec![1.0, 2.0, 3.0]));
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
        /// let layer = LinearLayer::new(Series::new("weights", vec![1.0, 2.0, 3.0]), Series::new("biases", vec![1.0, 2.0, 3.0]));
        /// ```
        ///
        /// # Arguments
        ///
        /// * `weights` - A Series of weights for the layer.
        ///
        /// * `biases` - A Series of biases for the layer.
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

        fn backward(&self, inputs: Series, grad_outputs: Series, lr: f64) -> (Series, Series) {
            // Calculate gradient with respect to inputs
            let grad_input: Series = &inputs * &grad_outputs;

            // Calculate gradient with respect to weights
            let grad_weights: Series = &inputs * &grad_outputs;

            let grad_biases: f64 = grad_outputs.f64().unwrap().sum().unwrap();
            
            let new_weights: Series = self.weights.clone() - grad_weights * lr;
            let new_biases: Series = self.biases.clone() - Series::new("biases", vec![grad_biases; self.biases.len()]) * lr;

            (new_weights, new_biases)
        }
    }
}