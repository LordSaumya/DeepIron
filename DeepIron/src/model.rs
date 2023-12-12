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
    use polars::{series::Series, frame::DataFrame};
    use polars::prelude::*;
    use crate::data_loader::DataFrameTransformer;

    /// Enum of supported loss functions.
    pub enum LossFunctionType {
        /// Mean squared error loss function.
        MeanSquaredError,
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

        fn intercept_gradient(&self, y: &Series, y_pred: &Series) -> f64 ;
    }

    impl LossFunction for LossFunctionType {
        /// Compute the mean squared error loss between the predicted and actual values.
        ///
        /// # Arguments
        ///
        /// * `y` - The actual values.
        ///
        /// * `y_pred` - The predicted values.
        ///
        /// # Returns
        ///
        /// * `f64` - The mean squared error.
        fn loss(&self, y: &Series, y_pred: &Series) -> f64 {
            match self {
                LossFunctionType::MeanSquaredError => {
                    let diff: Series = y - y_pred;
                    let squared_diff: Series = &diff * &diff;
                    squared_diff.mean().unwrap()
                },
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
                LossFunctionType::MeanSquaredError => {
                    let diff: Series = y - y_pred;
                    let mut gradients: Vec<f64> = Vec::with_capacity(x.width());
                    for i in 0..x.width() {
                        let feature_values: Series = x.get_col_by_index(i).unwrap();
                        let gradient: Series = &diff * &feature_values;
                        gradients.push(gradient.mean().unwrap() * -2.0);
                    }
                    Series::new("gradients", gradients)
                },
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
                LossFunctionType::MeanSquaredError => {
                    (y - y_pred).mean().unwrap() * -2.0
                },
            }
        }
    }
}
