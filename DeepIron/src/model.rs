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
pub mod Model {
    use polars::prelude::*;
    use polars::series::Series;

    /// An enumeration of the types of supported models.
    pub enum ModelType {
        Linear,
    }

    /// A trait that defines a model's fit and predict functions.
    pub trait Modeller {
        fn fit(&mut self, numEpochs: u32, learningRate: f64) -> Result<(), PolarsError>;
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
pub mod LossFunctions {
    use polars::series::Series;

    /// Enum of supported loss functions.
    pub enum LossFunctionType {
        MeanSquaredError,
    }

    impl LossFunctionType {
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
        pub fn loss(&self, y: &Series, y_pred: &Series) -> f64 {
            match self {
                LossFunctionType::MeanSquaredError => {
                    let diff: Series = y - y_pred;
                    let squared_diff: Series = &diff * &diff;
                    squared_diff.mean().unwrap()
                },
            }
        }

        pub fn gradient(&self, y: &Series, y_pred: &Series) -> Series {
            match self {
                LossFunctionType::MeanSquaredError => {
                    let diff: Series = y - y_pred;
                    diff * -2.0
                }
            }
        }
    }

    /// A trait that defines a loss function.
    ///
    /// # Example
    ///
    /// ```
    /// let loss = lossFunctions::MeanSquaredError;
    /// ```
    pub trait LossFunction {
        fn loss(y: &Series, y_pred: &Series) -> f64;
        fn gradient(y: &Series, y_pred: &Series) -> Series;
    }
}
