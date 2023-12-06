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
        Linear,
    }

    /// A trait that defines a model's fit and predict functions.
    pub trait Modeller {
        fn fit(&mut self, num_epochs: u32, learning_rate: f64) -> Result<(), PolarsError>;
        fn predict(&self, x: &DataFrame) -> Result<Series, PolarsError>;
    }
}

/// A set of loss functions for use in training models.
///
/// # Example
///
/// ```
/// let loss = lossFunctions::MeanSquaredError;
/// ```
pub mod loss_functions {
    use polars::{series::Series, frame::DataFrame};
    use polars::prelude::*;
    use crate::data_loader::DataFrameTransformer;

    /// Enum of supported loss functions.
    pub enum LossFunctionType {
        MeanSquaredError,
    }

    /// A trait that defines a loss function.
    ///
    /// # Example
    ///
    /// ```
    /// let loss = lossFunctions::MeanSquaredError;
    /// ```
    pub trait LossFunction {
        fn loss(&self, y: &Series, y_pred: &Series) -> f64;
        fn gradient(&self, x: &DataFrame, y: &Series, y_pred: &Series) -> Series;
    }

    impl LossFunction for LossFunctionType {
        /// Compute the mean squared error between the predicted and actual values.
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
    }
}
