/// A set of structs and functions that define a generic model.
///
/// # Example
///
/// ```
/// let model = Model::Linear::new();
///
/// model.fit(100, 0.01);
///
/// let y_pred = model.predict(&x);
///
/// ```
pub mod model {
    use polars::prelude::*;
    use polars::series::Series;

    /// An enumeration of the types of supported models.
    pub enum ModelType {
        /// Linear regression model
        Linear,
    }

    /// A trait that defines a model's fit and predict functions.
    pub trait Modeller {
        /// Fit the model to the training data.
        ///
        /// # Arguments
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
        /// model.fit(100, 0.01);
        ///
        /// ```
        fn fit(&mut self, num_epochs: u32, learning_rate: f64) -> Result<(), PolarsError>;

        /// Predict the target values for the given features.
        ///
        /// # Arguments
        ///
        /// * `x` - The features to predict the target values for.
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
        /// * `y` - The actual values.
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
        /// * `y` - The actual values.
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
    pub enum LossFunctionType {
        /// Mean squared error loss function.
        MeanSquaredError,
        /// Binary cross entropy loss function.
        BinaryCrossEntropy,
        /// Hinge loss function.
        Hinge,
    }

    /// A trait that defines a loss function.
    ///
    /// # Example
    ///
    /// ```
    /// let loss = loss_functions::MeanSquaredError;
    /// ```
    pub trait LossFunction {
        /// Compute the loss between the predicted and actual values.
        ///
        /// # Arguments
        ///
        /// * `y` - The actual values.
        ///
        /// * `y_pred` - The predicted values.
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
        /// * `x` - The features.
        ///
        /// * `y` - The actual values.
        ///
        /// * `y_pred` - The predicted values.
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
        fn gradient(&self, x: &DataFrame, y: &Series, y_pred: &Series) -> Series;

        fn intercept_gradient(&self, y: &Series, y_pred: &Series) -> f64;
    }

    impl LossFunction for LossFunctionType {
        /// Compute the loss between the predicted and actual values.
        ///
        /// # Arguments
        ///
        /// * `y` - The actual values.
        ///
        /// * `y_pred` - The predicted values.
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
                        let y_i = y_i.unwrap();
                        let y_pred_i = y_pred_i.unwrap();
                        loss += y_i * y_pred_i.ln() + (1.0 - y_i) * (1.0 - y_pred_i).ln();
                    }
                    loss = -loss / y.len() as f64;

                    loss
                }
                LossFunctionType::Hinge => {
                    // loss = 1/n * sum(max(0, 1 - y * y_pred))
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

        /// Compute the gradient of the mean squared error loss function.
        ///
        /// # Arguments
        ///
        /// * `x` - The features.
        ///
        /// * `y` - The actual values.
        ///
        /// * `y_pred` - The predicted values.
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
                LossFunctionType::Hinge => {
                    let mut gradients: Vec<f64> = Vec::with_capacity(x.width());
                    let y: &ChunkedArray<Float64Type> = y.f64().unwrap();
                    let y_pred: &ChunkedArray<Float64Type> = y_pred.f64().unwrap();

                    for i in 0..x.width() {
                        let feature_values: Series = x.get_col_by_index(i).unwrap();
                        let margin: f64 = y.get(i).unwrap() * y_pred.get(i).unwrap() * -1.0 + 1.0;
                        if margin > 0.0 {
                            let gradient: Series = &feature_values * (y.get(i).unwrap() * -1.0);
                            gradients.push(gradient.sum().unwrap());
                        } else {
                            gradients.push(0.0);
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
        /// * `y` - The actual values.
        ///
        /// * `y_pred` - The predicted values.
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
                LossFunctionType::MeanSquaredError => (y - y_pred).mean().unwrap() * -2.0,
                LossFunctionType::BinaryCrossEntropy => (y_pred - y).mean().unwrap() * 2.0,
                LossFunctionType::Hinge => {
                    let mut sum: f64 = 0.0;
                    let y: &ChunkedArray<Float64Type> = y.f64().unwrap();
                    let y_pred: &ChunkedArray<Float64Type> = y_pred.f64().unwrap();
                    for i in 0..y.len() {
                        // Check if the margin is violated for this data point
                        let margin: f64 = 1.0 - y.get(i).unwrap() * y_pred.get(i).unwrap();
                
                        // Add the margin violation (negative for misclassified points)
                        if margin > 0.0 {
                            sum += y.get(i).unwrap();
                        }
                    }
                    -sum / y.len() as f64 // Average the sum across data points
                }
            }
        }
    }
}
