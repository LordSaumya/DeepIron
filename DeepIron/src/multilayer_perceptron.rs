//! A module encapsulating the structures and methods for building a multilayer perceptron.
use crate::data_loader::DataFrameTransformer;
use crate::layers::layer::{Layer, LinearLayer};
use crate::model::activation_functions::{ActivationFunction, ActivationFunctionType};
use crate::model::kernel_functions::{KernelFunction, KernelFunctionType};
use crate::model::loss_functions::{LossFunction, LossFunctionType};
use crate::model::*;
use polars::prelude::*;
use polars::series::Series;

/// A struct that defines a multilayer perceptron.
///
/// # Example
///
/// ```
/// let model = Model::MLP::new(LossFunctionType::MeanSquaredError);
///
/// model.fit(&x, &y, num_epochs, learning_rate);
///
/// let y_pred = model.predict(&x);
///
/// ```
pub struct MLP {
    /// The loss function to use for the MLP.
    pub loss_function: LossFunctionType,
    /// The activation functions to use for the MLP.
    pub activation_functions: Vec<ActivationFunctionType>,
    /// The layers to use for the MLP.
    pub layers: Vec<LinearLayer>,
}

impl MLP {
    /// Create a new MLP model.
    ///
    /// # Example
    ///
    /// ```
    /// let model = Model::MLP::new(LossFunctionType::MeanSquaredError);
    /// ```
    pub fn new(loss_function: LossFunctionType) -> MLP {
        MLP {
            loss_function,
            activation_functions: Vec::new(),
            layers: Vec::new(),
        }
    }

    /// Create a new MLP model with the given layers.
    ///
    /// # Example
    ///
    /// ```
    /// let layers = vec![LinearLayer::new_random(32, [-1, 1]), LinearLayer::new_random(32, [-1, 1])];
    /// let activation_functions = vec![ActivationFunctionType::ReLU, ActivationFunctionType::ReLU];
    /// let loss_function = LossFunctionType::MeanSquaredError;
    /// 
    /// let model = Model::MLP::new_with_layers(loss_function, layers, activation_functions);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `loss_function` - The loss function to use for the MLP.
    ///
    /// * `layers` - A vector of linear layers to use for the MLP.
    ///
    /// * `activation_functions` - A vector of activation functions to use for the MLP.
    ///
    /// # Returns
    ///
    /// A new MLP with the given layers.
    pub fn new_with_layers(
        loss_function: LossFunctionType,
        layers: Vec<LinearLayer>,
        activation_functions: Vec<ActivationFunctionType>,
    ) -> MLP {
        assert_eq!(layers.len(), activation_functions.len());
        MLP {
            loss_function,
            activation_functions,
            layers,
        }
    }

    /// Add a layer with its activation function to the MLP.
    ///
    /// # Example
    ///
    /// ```
    /// let model = Model::MLP::new(LossFunctionType::MeanSquaredError);
    /// let layer = LinearLayer::new_random(32, [-1, 1]);
    ///
    /// let new_model = model.add_layer(layer, ActivationFunctionType::ReLU);
    ///
    /// ```
    ///
    /// # Arguments
    ///
    /// * `layer` - A linear layer to add to the MLP.
    ///
    /// * `activation_function` - The activation function to use for the layer.
    ///
    /// # Returns
    ///
    /// A new MLP with the given layer added.
    pub fn add_layer(
        &self,
        layer: LinearLayer,
        activation_function: ActivationFunctionType,
    ) -> MLP {
        let mut new_layers: Vec<LinearLayer> = self.layers.clone();
        let mut new_activation_functions: Vec<ActivationFunctionType> =
            self.activation_functions.clone();
        new_layers.push(layer);
        new_activation_functions.push(activation_function);
        MLP {
            loss_function: self.loss_function.clone(),
            activation_functions: new_activation_functions,
            layers: new_layers,
        }
    }

    /// Changes the ith layer of the MLP to the given layer.
    ///
    /// # Example
    ///
    /// ```
    /// let model = Model::MLP::new(LossFunctionType::MeanSquaredError);
    ///
    /// let new_model = model.set_layer(0, layer, ActivationFunctionType::ReLU);
    ///
    /// ```
    ///
    /// # Arguments
    ///
    /// * `i` - The index of the layer to change.
    ///
    /// * `layer` - A linear layer to add to the MLP.
    ///
    /// * `activation_function` - The activation function to use for the layer.
    ///
    /// # Returns
    ///
    /// A new MLP with the changed layer.
    pub fn set_layer(
        &self,
        i: usize,
        layer: LinearLayer,
        activation_function: ActivationFunctionType,
    ) -> MLP {

        if i >= self.layers.len() {
            panic!("Index out of bounds");
        }

        let mut new_layers: Vec<LinearLayer> = self.layers.clone();
        let mut new_activation_functions: Vec<ActivationFunctionType> =
            self.activation_functions.clone();
        new_layers[i] = layer;
        new_activation_functions[i] = activation_function;
        MLP {
            loss_function: self.loss_function.clone(),
            activation_functions: new_activation_functions,
            layers: new_layers,
        }
    }

    /// Perform forward propagation on the MLP with the given inputs.
    ///
    /// # Example
    ///
    /// ```
    /// let model = Model::MLP::new(LossFunctionType::MeanSquaredError);
    ///
    /// let y_pred = model.forward(&x);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `inputs` - A Series of inputs to the MLP.
    ///
    /// # Returns
    ///
    /// A Series of outputs from the MLP.
    pub fn forward(&self, inputs: Series) -> Series {
        let mut outputs: Series = inputs.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            outputs = layer.forward(outputs, self.activation_functions[i].clone());
        }
        outputs
    }

    /// Perform backward propagation on the MLP with the given inputs and outputs.
    ///
    /// # Example
    ///
    /// ```
    /// let model = Model::MLP::new(LossFunctionType::MeanSquaredError);
    /// 
    /// let y_pred = model.forward(&x);
    /// 
    /// let (updated_weights, updated_biases) = model.backward(&x, &y_pred, &y);
    ///
    /// ```
    ///
    /// # Arguments
    ///
    /// * `inputs` - A Series of inputs to the MLP.
    ///
    /// * `outputs` - A Series of outputs from the MLP.
    /// 
    /// * `true_values` - A Series of true values to compare the outputs to.
    ///
    /// # Returns
    ///
    /// A tuple of two vectors, each with a series of updated weights and biases for each layer.
    pub fn backward(
        &mut self,
        inputs: &Series,
        outputs: &Series,
        true_values: &Series,
        learning_rate: f64
    ) -> (Vec<Series>, Vec<Series>) {
        // Calculate the initial gradient of the loss with respect to the output
        let grad_outputs: f64 = self.loss_function.intercept_gradient(true_values, outputs);
    
        let mut weights_gradients: Vec<Series> = Vec::new();
        let mut biases_gradients: Vec<Series> = Vec::new();
    
        let mut grad_outputs_series = Series::new("grad_outputs", vec![grad_outputs]);
    
        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];
    
            let layer_inputs = if i == 0 {
                inputs.clone()
            } else {
                self.layers[i - 1].forward(inputs.clone(), self.activation_functions[i - 1].clone())
            };
    
            // Perform backward propagation for the current layer
            let (new_weights, new_biases) = layer.backward(layer_inputs.clone(), grad_outputs_series.clone(), learning_rate);
    
            // Store the updated weights and biases
            weights_gradients.push(new_weights.clone());
            biases_gradients.push(new_biases.clone());
    
            // Update the gradient for the next layer (moving backwards)
            grad_outputs_series = new_weights.clone();
        }
    
        // Reverse the gradients to match the order of the layers
        weights_gradients.reverse();
        biases_gradients.reverse();
    
        (weights_gradients, biases_gradients)
    }
}

impl model::SupervisedModeller for MLP {
    /// Fit the model to the data.
    ///
    /// # Example
    ///
    /// ```
    ///
    /// let model = MLP::new(LossFunctionType::MeanSquaredError);
    ///
    /// model.fit(&x, &y, num_epochs, learning_rate);
    ///
    /// ```
    fn fit(
        &mut self,
        x: &DataFrame,
        y: &Series,
        num_epochs: u32,
        learning_rate: f64,
    ) -> Result<(), PolarsError> {
        let x: Series = x.get_col_by_index(0)?;
        for _ in 0..num_epochs {
            let y_pred: Series = self.forward(x.clone());
            let (updated_weights, updated_biases) = self.backward(&x, &y_pred, y, learning_rate);
            for i in 0..self.layers.len() {
                self.layers[i].weights = updated_weights[i].clone();
                self.layers[i].biases = updated_biases[i].clone();
            }
        }
        Ok(())
    }

    /// Predict the output of the model given the input.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let model = MLP::new(LossFunctionType::MeanSquaredError);
    /// 
    /// 
    fn predict(&self, x: &DataFrame) -> Result<Series, PolarsError> {
        let mut outputs: Series = x.get_col_by_index(0).unwrap();
        for (i, layer) in self.layers.iter().enumerate() {
            outputs = layer.forward(outputs, self.activation_functions[i].clone());
        }
        Ok(outputs)
    }

    /// Calculate the loss of the model given the input and output.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let model = MLP::new(LossFunctionType::MeanSquaredError);
    /// 
    /// let loss = model.loss(&x, &y);
    /// 
    /// ```
    fn loss(&self, x: &DataFrame, y: &Series) -> Result<f64, PolarsError> {
        let y_pred: Series = self.predict(x)?;
        Ok(self.loss_function.loss(&y_pred, y))
    }

    /// Calculate the accuracy of the model given the input and output.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let model = MLP::new(LossFunctionType::MeanSquaredError);
    /// 
    /// let accuracy = model.accuracy(&x, &y);
    /// 
    /// ```
    fn accuracy(&self, x: &DataFrame, y: &Series) -> Result<f64, PolarsError> {
        let y_pred: Series = self.predict(x)?;
        let accuracy: f64 = y_pred.equal(y).unwrap().sum().unwrap() as f64 / y_pred.len() as f64;
        Ok(accuracy)
    }
}

/// Implement the Display trait for printing an MLP.
impl std::fmt::Display for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut layers_string = String::new();
        for (i, layer) in self.layers.iter().enumerate() {
            layers_string.push_str(&format!(
                "Layer {}:\n\t\tNumber of neurons: {}\n",
                i,
                layer.weights.len()
            ));
            layers_string.push_str(&format!(
                "\t\tActivation Function: {}",
                self.activation_functions[i]
            ));
        }

        write!(
            f,
            "MLP - {} layers\n
        Loss Function: {}\n
        Layers: {} \n 
        ",
            self.layers.len(),
            self.loss_function,
            layers_string
        )
    }
}
