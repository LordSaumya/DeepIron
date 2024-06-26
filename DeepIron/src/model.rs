//! A set of submodules used to define a generic model.

/// A set of structs and functions that define a generic model.
///
/// # Example
///
/// ```
/// let model = Model::Linear::new();
///
/// model.fit(&x, &y, 100, 0.01);
///
/// let y_pred = model.predict(&x);
///
/// ```
pub mod model {
    use polars::prelude::*;
    use polars::series::Series;

    /// A trait that defines a supervised learning model's fit and predict functions.
    pub trait SupervisedModeller {
        /// Fit the model to the training data.
        ///
        /// # Arguments
        ///
        /// * `x` - The Dataframe containing the features for training.
        ///
        /// * `y` - The Series of expected target values for training.
        ///
        /// * `num_epochs` - The number of epochs to train for.
        ///
        /// * `learning_rate` - The initial learning rate to use during training.
        ///
        /// # Returns
        ///
        /// * `Result<(), PolarsError>` - A result indicating if the model was fit successfully.
        ///
        /// # Example
        ///
        /// ```no_run
        /// let model = Model::Linear::new();
        ///
        /// model.fit(&x, &y, 100, 0.01);
        ///
        /// ```
        fn fit(
            &mut self,
            x: &DataFrame,
            y: &Series,
            num_epochs: u32,
            learning_rate: f64,
        ) -> Result<(), PolarsError>;

        /// Predict the target values for the given features.
        ///
        /// # Arguments
        ///
        /// * `x` - The DataFrame containing the features to predict for.
        ///
        /// # Returns
        ///
        /// * `Result<Series, PolarsError>` - A result containing the predicted target values.
        ///
        /// # Example
        ///
        /// ```no_run
        ///
        /// let y_pred = model.predict(&x);
        ///
        /// ```
        fn predict(&self, x: &DataFrame) -> Result<Series, PolarsError>;

        /// Compute the accuracy of the model.
        ///
        /// # Arguments
        ///
        /// * `x` - The features.
        ///
        /// * `y` - The true values.
        ///
        /// # Returns
        ///
        /// * `Result<f64, PolarsError>` - A result containing the accuracy.
        ///
        /// # Example
        ///
        /// ```no_run
        ///
        /// let accuracy = model.accuracy(&x, &y);
        ///
        /// ```
        fn accuracy(&self, x: &DataFrame, y: &Series) -> Result<f64, PolarsError>;

        /// Compute the loss of the model using its stored loss function.
        ///
        /// # Arguments
        ///
        /// * `x` - The features.
        ///
        /// * `y` - The true values.
        ///
        /// # Returns
        ///
        /// * `Result<f64, PolarsError>` - A result containing the loss.
        ///
        /// # Example
        ///
        /// ```no_run
        ///
        /// let loss = model.loss(&x, &y);
        ///
        /// ```
        fn loss(&self, x: &DataFrame, y: &Series) -> Result<f64, PolarsError>;
    }

    /// A trait that defines a clustering model's fit, predict, and evaluation functions.
    pub trait ClusterModeller {
        /// Fit the model to the training data.
        ///
        /// # Arguments
        ///
        /// * `x` - The Dataframe containing the features for training.
        ///
        /// # Returns
        ///
        /// * `Result<(), PolarsError>` - A result indicating if the model was fit successfully.
        ///
        /// # Example
        ///
        /// ```no_run
        /// let model = Model::KMeans::new_random(3, EndCondition::MaxIter(100));
        ///
        /// model.fit(&x);
        ///
        /// ```
        fn fit(&mut self, x: &DataFrame) -> Result<(), PolarsError>;

        /// Predict the cluster assignments for the given features.
        ///
        /// # Arguments
        ///
        /// * `x` - The DataFrame containing the features to predict for.
        ///
        /// # Returns
        ///
        /// * `Result<Series, PolarsError>` - A result containing the predicted cluster assignments.
        ///
        /// # Example
        ///
        /// ```no_run
        ///
        /// let cluster_assignments = model.predict(&x);
        ///
        /// ```
        fn predict(&mut self, x: &DataFrame) -> Result<Series, PolarsError>;

        /// Computes the compactness of the clusters.
        /// 
        /// # Arguments
        /// 
        /// * `x` - The DataFrame containing the features to evaluate the model on.
        /// 
        /// # Returns
        /// 
        /// * `Result<f64, PolarsError>` - A result containing the compactness of the clusters.
        /// 
        /// # Example
        /// 
        /// ```no_run
        /// 
        /// let compactness = model.compactness(&x);
        /// 
        /// ```
        fn compactness(&mut self, x: &DataFrame) -> Result<f64, PolarsError>;
    }
}


/// A set of loss functions for use in training models.
///
/// # Example
///
/// ```
/// let loss = loss_functions::MeanSquaredError;
/// ```
pub mod loss_functions {
    use crate::data_loader::DataFrameTransformer;
    use polars::prelude::*;
    use polars::{frame::DataFrame, series::Series};

    /// Enum of supported loss functions.
    #[derive(Clone, PartialEq)]
    pub enum LossFunctionType {
        /// Mean squared error loss function.
        MeanSquaredError,
        /// Binary cross entropy loss function.
        BinaryCrossEntropy,
        /// Hinge loss function.
        Hinge,
    }

    /// Implement the Display trait for printing LossFunctionType.
    impl std::fmt::Display for LossFunctionType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                LossFunctionType::MeanSquaredError => write!(f, "Mean Squared Error"),
                LossFunctionType::BinaryCrossEntropy => write!(f, "Binary Cross Entropy"),
                LossFunctionType::Hinge => write!(f, "Hinge"),
            }
        }
    }

    /// A trait that defines a loss function.
    ///
    /// # Example
    ///
    /// ```
    /// let loss = loss_functions::MeanSquaredError;
    /// ```
    pub trait LossFunction {
        /// Compute the loss between the predicted and true values.
        ///
        /// # Arguments
        ///
        /// * `y` - The Series of true values.
        ///
        /// * `y_pred` - The Series of predicted values.
        ///
        /// # Returns
        ///
        /// * `f64` - The loss.
        ///
        /// # Example
        ///
        /// ```
        /// let lossFn = loss_functions::MeanSquaredError;
        ///
        /// let loss_value = lossFn.loss(&y, &y_pred);
        ///
        /// ```
        fn loss(&self, y: &Series, y_pred: &Series) -> f64;

        /// Compute the gradient of the loss function.
        ///
        /// # Arguments
        ///
        /// * `x` - The DataFrame of features.
        ///
        /// * `y` - The Series of true values.
        ///
        /// * `y_pred` - The Series of predicted values.
        ///
        /// # Returns
        ///
        /// * `Series` - The gradients.
        ///
        /// # Example
        ///
        /// ```
        /// let lossFn = loss_functions::MeanSquaredError;
        ///
        /// let gradient = lossFn.gradient(&x, &y, &y_pred);
        ///
        /// ```
        fn gradient(&self, x: &DataFrame, y: &Series, y_pred: &Series) -> Series;

        /// Compute the gradient of the loss function for the intercept (does not take into account individual features values).
        ///
        /// # Arguments
        ///
        /// * `y` - The Series of true values.
        ///
        /// * `y_pred` - The Series of predicted values.
        ///
        /// # Returns
        ///
        /// * `f64` - The gradient.
        ///
        /// # Example
        ///
        /// ```
        ///
        /// let gradient = lossFn.intercept_gradient(&y, &y_pred);
        ///
        /// ```
        fn intercept_gradient(&self, y: &Series, y_pred: &Series) -> f64;
    }

    impl LossFunction for LossFunctionType {
        /// Compute the loss between the predicted and true values.
        ///
        /// # Arguments
        ///
        /// * `y` - The Series of true values.
        ///
        /// * `y_pred` - The Series of predicted values.
        ///
        /// # Returns
        ///
        /// * `f64` - The error.
        fn loss(&self, y: &Series, y_pred: &Series) -> f64 {
            match self {
                LossFunctionType::MeanSquaredError => {
                    // loss = 1/n * sum((y - y_pred)^2)
                    let diff: Series = y - y_pred;
                    let squared_diff: Series = &diff * &diff;
                    squared_diff.mean().unwrap()
                }
                LossFunctionType::BinaryCrossEntropy => {
                    // loss = -1/n * sum(y * log(y_pred) + (1 - y) * log(1 - y_pred))
                    let mut loss: f64 = 0.0;
                    let y: &ChunkedArray<Float64Type> = y.f64().unwrap();
                    let y_pred: &ChunkedArray<Float64Type> = y_pred.f64().unwrap();

                    for (y_i, y_pred_i) in y.into_iter().zip(y_pred.into_iter()) {
                        let y_i: f64 = y_i.unwrap();
                        let y_pred_i: f64 = y_pred_i.unwrap();
                        loss += y_i * y_pred_i.ln() + (1.0 - y_i) * (1.0 - y_pred_i).ln();
                    }
                    loss = -loss / y.len() as f64;

                    loss
                }
                LossFunctionType::Hinge => {
                    // loss = sum(max(0, 1 - y * y_pred))
                    let mut loss: f64 = 0.0;
                    let y: &ChunkedArray<Float64Type> = y.f64().unwrap();
                    let y_pred: &ChunkedArray<Float64Type> = y_pred.f64().unwrap();

                    for (y_i, y_pred_i) in y.into_iter().zip(y_pred.into_iter()) {
                        let y_i: f64 = y_i.unwrap();
                        let y_pred_i: f64 = y_pred_i.unwrap();
                        loss += (1.0 - y_i * y_pred_i).max(0.0);
                    }

                    loss
                }
            }
        }

        /// Compute the gradient of the loss function with respect to each feature.
        ///
        /// # Arguments
        ///
        /// * `x` - The DataFrame of features.
        ///
        /// * `y` - The Series of true values.
        ///
        /// * `y_pred` - The Series of predicted values.
        ///
        /// # Returns
        ///
        /// * `Series` - The gradient.
        ///
        /// # Example
        ///
        /// ```
        /// let lossFn = loss_functions::MeanSquaredError;
        ///
        /// let gradient = lossFn.gradient(&x, &y, &y_pred);
        ///
        /// ```
        fn gradient(&self, x: &DataFrame, y: &Series, y_pred: &Series) -> Series {
            match self {
                // gradient = 1/n * sum(2 * (y_pred - y) * x)
                LossFunctionType::MeanSquaredError => {
                    let diff: Series = y - y_pred;
                    let mut gradients: Vec<f64> = Vec::with_capacity(x.width());
                    for i in 0..x.width() {
                        let feature_values: Series = x.get_col_by_index(i).unwrap();
                        let gradient: Series = &diff * &feature_values;
                        gradients.push(gradient.mean().unwrap() * -2.0);
                    }
                    Series::new("gradients", gradients)
                }
                // gradient = 1/n * sum((y_pred - y) * x)
                LossFunctionType::BinaryCrossEntropy => {
                    let mut gradients: Vec<f64> = Vec::with_capacity(x.width());
                    for i in 0..x.width() {
                        let feature_values: Series = x.get_col_by_index(i).unwrap();
                        let gradient: Series = &feature_values * &(y_pred - y);
                        gradients.push(gradient.mean().unwrap());
                    }
                    Series::new("gradients", gradients)
                }
                // gradient = sum(-1 * x * y) if margin > 0 else 0
                LossFunctionType::Hinge => {
                    let mut gradients: Vec<f64> = Vec::with_capacity(x.width());
                    let y: &ChunkedArray<Float64Type> = y.f64().unwrap();
                    let y_pred: &ChunkedArray<Float64Type> = y_pred.f64().unwrap();

                    for (y_i, feature_col) in y.into_iter().zip(y_pred.into_iter()).zip(x.iter()) {
                        let margin: f64 = y_i.0.unwrap() - y_i.1.unwrap();
                        let derivative: f64 = if margin > 0.0 { -1.0 } else { 0.0 };
                        let feature_col: &ChunkedArray<Float64Type> = feature_col.f64().unwrap();
                        for j in 0..feature_col.len() {
                            gradients.push(derivative * feature_col.get(j).unwrap());
                        }
                    }
                    Series::new("gradients", gradients)
                }
            }
        }

        /// Compute the gradient of the loss function for the intercept (does not take into account individual features values).
        ///
        /// # Arguments
        ///
        /// * `y` - The Series of true values.
        ///
        /// * `y_pred` - The Series of predicted values.
        ///
        /// # Returns
        ///
        /// * `f64` - The gradient.
        ///
        /// # Example
        ///
        /// ```
        ///
        /// let gradient = lossFn.intercept_gradient(&y, &y_pred);
        ///
        /// ```
        fn intercept_gradient(&self, y: &Series, y_pred: &Series) -> f64 {
            match self {
                // gradient = 1/n * -2 * sum((y_pred - y))
                LossFunctionType::MeanSquaredError => (y - y_pred).mean().unwrap() * -2.0,
                // gradient = 1/n * -2 * sum((y_pred - y))
                LossFunctionType::BinaryCrossEntropy => (y - y_pred).mean().unwrap() * -2.0,
                // gradient = sum(-1 * y) if margin > 0 else 0
                LossFunctionType::Hinge => {
                    let mut sum: f64 = 0.0;
                    let y: &ChunkedArray<Float64Type> = y.f64().unwrap();
                    let y_pred: &ChunkedArray<Float64Type> = y_pred.f64().unwrap();

                    for i in 0..y.len() {
                        // Calculate the correct margin difference
                        let margin = y.get(i).unwrap() - y_pred.get(i).unwrap();

                        // Add the hinge derivative (-1 for violated margins)
                        if margin > 0.0 {
                            sum -= 1.0; // Directly add -1 for violated margins
                        }
                    }

                    -sum / y.len() as f64 // Average the sum across data points
                }
            }
        }
    }
}

/// A set of activation functions for use in training models.
///
/// # Example
///
/// ```
///
/// let activation = activation_functions::Sigmoid;
///
/// ```
pub mod activation_functions {
    use polars::prelude::*;
    use polars::series::Series;

    /// Enum of supported activation functions.
    #[derive(Clone, PartialEq, Debug)]
    pub enum ActivationFunctionType {
        /// Identity activation function (does nothing)
        Identity,
        /// Sigmoid activation function.
        Sigmoid,
        /// Rectified linear unit activation function.
        ReLU,
        /// Softmax activation function.
        Softmax,
        /// Leaky ReLU activation function.
        LReLU(f64)
    }

    /// Implement the Display trait for printing ActivationFunctionType.
    impl std::fmt::Display for ActivationFunctionType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ActivationFunctionType::Identity => write!(f, "Identity"),
                ActivationFunctionType::Sigmoid => write!(f, "Sigmoid"),
                ActivationFunctionType::ReLU => write!(f, "ReLU"),
                ActivationFunctionType::Softmax => write!(f, "Softmax"),
                ActivationFunctionType::LReLU(alpha) => write!(f, "Leaky ReLU (alpha = {})", alpha),
            }
        }
    }

    /// A trait that defines an activation function.
    ///
    /// # Example
    ///
    /// ```
    ///
    /// let activation = activation_functions::Sigmoid;
    ///
    /// ```
    pub trait ActivationFunction {
        /// Compute the activation of the given values.
        ///
        /// # Arguments
        ///
        /// * `values` - The Series of values to compute the activation for.
        ///
        /// # Returns
        ///
        /// * `Series` - The Series of activation values.
        ///
        /// # Example
        ///
        /// ```
        ///
        /// let activated_values = activation.activate(&values);
        ///
        /// ```
        fn activate(&self, values: &Series) -> Series;

        /// Compute the gradient of the activation function.
        ///
        /// # Arguments
        ///
        /// * `values` - The Series of values to compute the gradient for.
        ///
        /// # Returns
        ///
        /// * `Series` - The Series of gradients.
        ///
        /// # Example
        ///
        /// ```
        ///
        /// let gradients = activation.gradient(&values);
        ///
        /// ```
        fn gradient(&self, values: &Series) -> Series;
    }

    impl ActivationFunction for ActivationFunctionType {
        /// Compute the activation of the given values.
        ///
        /// # Arguments
        ///
        /// * `values` - The Series of values to compute the activation for.
        ///
        /// # Returns
        ///
        /// * `Series` - The Series of activation values.
        ///
        /// # Example
        ///
        /// ```
        ///
        /// let activated_values = activation.activate(&values);
        ///
        /// ```
        fn activate(&self, values: &Series) -> Series {
            match self {
                ActivationFunctionType::Identity => {
                    values.clone().rename("activated_values").clone()
                }
                ActivationFunctionType::Sigmoid => {
                    // sigmoid(x) = 1 / (1 + e^-x)
                    values
                        .f64()
                        .unwrap()
                        .apply(|value| Some(1.0 / (1.0 + f64::exp(-value.unwrap()))))
                        .into_series()
                        .rename("activated_values")
                        .clone()
                }
                ActivationFunctionType::ReLU => {
                    // ReLU(x) = max(0, x)
                    values
                        .f64()
                        .unwrap()
                        .apply(|value| Some(value.unwrap().max(0.0)))
                        .into_series()
                        .rename("activated_values")
                        .clone()
                }
                ActivationFunctionType::Softmax => {
                    // softmax(x) = e^x / sum(e^x)
                    let activated_values: Series = values
                    .f64()
                    .unwrap()
                    .apply(|value| Some(f64::exp(value.unwrap())))
                    .into_series()
                    .clone();
                    let sum: f64 = activated_values.sum().unwrap();
                    activated_values
                        .f64()
                        .unwrap()
                        .apply(|value| Some(value.unwrap() / sum))
                        .into_series()
                        .rename("activated_values")
                        .clone()
                }
                ActivationFunctionType::LReLU(alpha) => {
                    // Leaky ReLU(x) = x if x > 0 else alpha * x
                    values
                        .f64()
                        .unwrap()
                        .apply(|value| Some(if value.unwrap() > 0.0 { value.unwrap() } else { alpha * value.unwrap() }))
                        .into_series()
                        .rename("activated_values")
                        .clone()
                }
            }
        }

        /// Compute the gradient of the activation function.
        /// 
        /// # Arguments
        /// 
        /// * `values` - The Series of values to compute the gradient for.
        /// 
        /// # Returns
        /// 
        /// * `Series` - The Series of gradients.
        /// 
        /// # Example
        /// 
        /// ```
        /// 
        /// let gradients = activation.gradient(&values);
        /// 
        /// ```
        fn gradient(&self, values: &Series) -> Series {
            match self {
                ActivationFunctionType::Identity => {
                    // identity'(x) = 1
                    Series::new("gradients", vec![1.0; values.len()])
                }
                ActivationFunctionType::Sigmoid => {
                    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                    let activated_values = self.activate(values);
                    let mut gradients: Vec<f64> = Vec::with_capacity(values.len());
                    for value in activated_values.f64().unwrap().into_iter() {
                        let value: f64 = value.unwrap();
                        gradients.push(value * (1.0 - value));
                    }
                    Series::new("gradients", gradients)
                }
                ActivationFunctionType::ReLU => {
                    // ReLU'(x) = 1 if x > 0 else 0
                    let mut gradients: Vec<f64> = Vec::with_capacity(values.len());
                    for value in values.f64().unwrap().into_iter() {
                        let value: f64 = value.unwrap();
                        gradients.push(if value > 0.0 { 1.0 } else { 0.0 });
                    }
                    Series::new("gradients", gradients)
                }
                ActivationFunctionType::Softmax => {
                    // Softmax gradient involves the Jacobian matrix of the softmax function
                    let values: Vec<f64> = values.f64().unwrap().into_iter().map(|v| v.unwrap()).collect();
                    let len: usize = values.len();
                
                    // Compute softmax probabilities
                    let max_value: f64 = values.iter().cloned().fold(f64::NAN, f64::max);
                    let exp_values: Vec<f64> = values.iter().map(|v| (v - max_value).exp()).collect();
                    let sum_exp_values: f64 = exp_values.iter().sum();
                    let softmax: Vec<f64> = exp_values.iter().map(|v| v / sum_exp_values).collect();
                
                    // Compute the Jacobian matrix of the softmax
                    let mut jacobian: Vec<Vec<f64>> = vec![vec![0.0; len]; len];
                    for i in 0..len {
                        for j in 0..len {
                            if i == j {
                                jacobian[i][j] = softmax[i] * (1.0 - softmax[i]);
                            } else {
                                jacobian[i][j] = -softmax[i] * softmax[j];
                            }
                        }
                    }
                
                    // Flatten the Jacobian matrix to a single vector
                    let gradients: Vec<f64> = jacobian.into_iter().flat_map(|row| row.into_iter()).collect();
                    Series::new("gradients", gradients)
                }
                ActivationFunctionType::LReLU(alpha) => {
                    // Leaky ReLU'(x) = 1 if x > 0 else alpha
                    let mut gradients: Vec<f64> = Vec::with_capacity(values.len());
                    for value in values.f64().unwrap().into_iter() {
                        let value: f64 = value.unwrap();
                        gradients.push(if value > 0.0 { 1.0 } else { *alpha });
                    }
                    Series::new("gradients", gradients)
                }
            }
        }
    }
}

/// A set of kernel functions for use in SVMs
///
/// # Example
///
/// ```
/// let kernel = kernel_functions::Linear;
/// ```
pub mod kernel_functions {
    use polars::prelude::*;
    use polars::series::Series;

    /// Enum of supported kernel functions.
    pub enum KernelFunctionType {
        /// Identity kernel function (does nothing)
        Identity,
        /// Linear kernel function.
        Linear,
        /// Polynomial kernel function with constant a and exponent b.
        Polynomial(f64, f64),
        /// Radial basis function kernel function with constant gamma.
        RadialBasisFunction(f64),
    }

    /// A trait that defines a kernel function.
    ///
    /// # Example
    ///
    /// ```
    /// let kernel = kernel_functions::Linear;
    /// ```
    pub trait KernelFunction {
        /// Compute the kernel of the given values.
        ///
        /// # Arguments
        ///
        /// * `x` - The first Series of values.
        ///
        /// * `y` - The second Series of values.
        ///
        /// # Returns
        ///
        /// * `Series` - The kernel values.
        ///
        /// # Example
        ///
        /// ```
        /// let kernel = kernel_functions::Linear;
        ///
        /// let kernel_value = kernel.kernel(&x, &y);
        ///
        /// ```
        fn kernel(&self, x: &Series, y: &Series) -> Series;
    }

    impl KernelFunction for KernelFunctionType {
        /// Compute the kernel of the given values.
        ///
        /// # Arguments
        ///
        /// * `x` - The first Series of values.
        ///
        /// * `y` - The second Series of values.
        ///
        /// # Returns
        ///
        /// * `Series` - The kernel values.
        ///
        /// # Example
        ///
        /// ```
        /// let kernel = kernel_functions::Linear;
        ///
        /// let kernel_value = kernel.kernel(&x, &y);
        ///
        /// ```
        fn kernel(&self, x: &Series, y: &Series) -> Series {
            match self {
                KernelFunctionType::Identity => x.clone().rename("kernel").clone(),
                KernelFunctionType::Linear => {
                    // kernel = x * y
                    let mut kernel: Vec<f64> = Vec::with_capacity(x.len());
                    for (x_i, y_i) in x
                        .f64()
                        .unwrap()
                        .into_iter()
                        .zip(y.f64().unwrap().into_iter())
                    {
                        let x_i: f64 = x_i.unwrap();
                        let y_i: f64 = y_i.unwrap();
                        kernel.push(x_i * y_i);
                    }

                    Series::new("kernel", kernel)
                }
                KernelFunctionType::Polynomial(a, b) => {
                    // kernel = (x * y + a)^b
                    let mut kernel: Vec<f64> = Vec::with_capacity(x.len());
                    for (x_i, y_i) in x
                        .f64()
                        .unwrap()
                        .into_iter()
                        .zip(y.f64().unwrap().into_iter())
                    {
                        let x_i: f64 = x_i.unwrap();
                        let y_i: f64 = y_i.unwrap();
                        kernel.push((x_i * y_i + a).powf(*b));
                    }

                    Series::new("kernel", kernel)
                }
                KernelFunctionType::RadialBasisFunction(gamma) => {
                    // kernel = e^(-gamma * ||x - y||^2)
                    let mut kernel: Vec<f64> = Vec::with_capacity(x.len());
                    for (x_i, y_i) in x
                        .f64()
                        .unwrap()
                        .into_iter()
                        .zip(y.f64().unwrap().into_iter())
                    {
                        let x_i: f64 = x_i.unwrap();
                        let y_i: f64 = y_i.unwrap();
                        kernel.push((-gamma * (x_i - y_i).powi(2)).exp());
                    }

                    Series::new("kernel", kernel)
                }
            }
        }
    }
}
