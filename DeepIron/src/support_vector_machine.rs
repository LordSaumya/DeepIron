use crate::data_loader::DataFrameTransformer;
use crate::model::loss_functions::{LossFunction, LossFunctionType};
use crate::model::*;
use polars::prelude::*;
use polars::series::Series;

/// A struct that defines a support vector machine.
/// 
/// # Example
/// 
/// ```
/// let model = Model::SVM::new();
///
/// model.fit(&x, &y);
/// 
/// let y_pred = model.predict(&x);
/// 
/// ```
pub struct SVM {
    // Fields for training
    pub x: DataFrame,
    pub y: Series,
    pub loss_function: LossFunctionType,

    // Fields to store results
    pub intercept: f64,
    pub coefficients: Vec<f64>,
}

impl SVM {
    /// Create a new SVM model.
    /// 
    /// # Example
    /// 
    /// ```
    /// let model = Model::SVM::new();
    /// ```
    pub fn new(x: DataFrame, y: Series) -> SVM {
        let x_width = x.width();
        SVM {
            x: x,
            y: y,
            loss_function: LossFunctionType::Hinge,
            intercept: 0.0,
            coefficients: vec![0.0; x_width],
        }
    }

    fn compute_gradients(&self, predictions: &Series) -> (f64, Vec<f64>) {
        let mut gradients: Vec<f64> = Vec::with_capacity(self.coefficients.len());
        let intercept_gradient: f64 = self.loss_function.intercept_gradient( &self.y, predictions);

        for (_i, _) in self.coefficients.iter().enumerate() {
            let gradient: f64 = self.loss_function.gradient(&self.x, &self.y, predictions).mean().unwrap();
            gradients.push(gradient);
        }

        (intercept_gradient, gradients)
    }
}

impl model::Modeller for SVM {
    /// Fit the model to the data.
    /// 
    /// # Example
    /// 
    /// ```
    /// let model = Model::SVM::new();
    /// 
    /// model.fit(&x, &y);
    /// ```
    fn fit(&mut self, num_epochs: u32, learning_rate: f64) -> Result<(), PolarsError> {
        // Check if data are valid
        if self.x.shape().0 != self.y.len() {
            return Err(PolarsError::ShapeMismatch("Shape mismatch between X and y".into()));
        }
        
        for _ in 0..num_epochs {
            let predictions: Series = self.predict(&self.x)?;
            let gradients: (f64, Vec<f64>) = self.compute_gradients(&predictions);

            self.intercept -= learning_rate * gradients.0;

            for (i, coefficient) in self.coefficients.iter_mut().enumerate() {
                *coefficient -= learning_rate * gradients.1[i];
            }
        }

        Ok(())
    }

    fn predict(&self, x: &DataFrame) -> Result<Series, PolarsError> {
        let mut predictions: Series = Series::new("prediction", vec![self.intercept; x.height()]);
        
        for (i, coefficient) in self.coefficients.iter().enumerate() {
            let x_column: &ChunkedArray<Float64Type> = x.get_col_by_index(i).unwrap().f64().unwrap();
            let prediction: Series = (x_column * *coefficient).into_series();
            predictions = predictions + prediction;
        }

        // Run activation function
        predictions = self.loss_function.activation(&predictions);

        Ok(predictions)
    }
}