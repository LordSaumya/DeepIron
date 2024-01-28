pub mod data_loader;
pub mod layers;
pub mod linear_regression;
pub mod logistic_regression;
pub mod model;
pub mod multilayer_perceptron;
pub mod support_vector_machine;

#[cfg(test)]
mod tests {
    use crate::layers::layer::LinearLayer;
    use crate::model::activation_functions;

    use self::model::loss_functions;

    use super::*;
    use data_loader::*;
    use linear_regression::*;
    use logistic_regression::*;
    use model::activation_functions::{ActivationFunction, ActivationFunctionType};
    use model::kernel_functions::{KernelFunction, KernelFunctionType};
    use model::loss_functions::{LossFunction, LossFunctionType};
    use model::model::Modeller;
    use multilayer_perceptron::*;
    use layers::layer::Layer;
    use polars::prelude::*;
    use std::path::Path;
    use support_vector_machine::*;

    #[test]
    fn test_load_csv() {
        // Load the CSV file
        let path: &Path = Path::new("test/loadTest.csv");
        let result: Result<DataFrame, PolarsError> = data_loader::data_loader_util::load_csv(path);

        println!("{:?}", result);

        // Check if the result is Ok and the DataFrame is not empty
        assert!(result.is_ok());
        let df: DataFrame = result.expect("Failed to load CSV file");
        assert!(!df.is_empty());
    }

    #[test]
    fn test_transform_by_col_custom_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();

        // Apply a transformation
        df = df.transform_cols(&["col1"], |s: &Series| s * 2).unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[2.0, 4.0, 6.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_transform_by_col_identity_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();

        // Apply a transformation
        df = df
            .transform_cols(&["col1"], transformer_functions::identity())
            .unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_transform_by_col_power_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();

        // Apply a transformation
        df = df
            .transform_cols(&["col1"], transformer_functions::power(2.0))
            .unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 4.0, 9.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_transform_by_col_log_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 4.0])]).unwrap();

        // Apply a transformation
        df = df
            .transform_cols(&["col1"], transformer_functions::log(2.0))
            .unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[0.0, 1.0, 2.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_split() {
        // Create a simple DataFrame for testing
        let df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0, 4.0, 5.0])]).unwrap();

        // Split the DataFrame
        let (train, test) = df.split(0.8).unwrap();

        // Check if the split is done correctly
        assert_eq!(train.height(), 4);
        assert_eq!(test.height(), 1);
    }

    #[test]
    fn test_z_norm_cols() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0, 4.0, 5.0])]).unwrap();

        // Z-normalise the column
        df = df.z_norm_cols(&["col1"]).unwrap();

        let std: f64 = f64::sqrt(2.0);
        let mean: f64 = 3.0;

        // Check if the z-normalisation is done correctly
        let expected_result: DataFrame = DataFrame::new(vec![Series::new(
            "col1",
            &[
                (1.0 - mean) / std,
                (2.0 - mean) / std,
                (3.0 - mean) / std,
                (4.0 - mean) / std,
                (5.0 - mean) / std,
            ],
        )])
        .unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_chained_transformations() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 4.0])]).unwrap();

        // Apply a transformation
        df = df
            .transform_cols(&["col1"], transformer_functions::power(2.0))
            .transform_cols(&["col1"], transformer_functions::log(2.0))
            .transform_cols(&["col1"], transformer_functions::identity())
            .z_norm_cols(&["col1"])
            .unwrap();

        let std: f64 = f64::sqrt(8.0 / 3.0);
        let mean: f64 = 2.0;

        // Check if the transformation is applied correctly
        let expected_result = DataFrame::new(vec![Series::new(
            "col1",
            &[(0.0 - mean) / std, (2.0 - mean) / std, (4.0 - mean) / std],
        )])
        .unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_min_max_norm_cols() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0, 4.0, 5.0])]).unwrap();

        // Min-max normalise the column
        df = df.min_max_norm_cols(&["col1"]).unwrap();

        // Check if the min-max normalisation is done correctly
        let expected_result: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[0.0, 0.25, 0.5, 0.75, 1.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_mean_squared_error_loss_zero() {
        // Create a simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);

        // Compute the mean squared error
        let mse: f64 = LossFunctionType::MeanSquaredError.loss(&x, &x);

        // Check if the mean squared error is computed correctly
        assert_eq!(mse, 0.0);
    }

    #[test]
    fn test_mean_squared_error_loss_non_zero() {
        // Create two simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);
        let y: Series = Series::new("y", &[4.0, 5.0, 6.0]);

        // Compute the mean squared error
        let mse = LossFunctionType::MeanSquaredError.loss(&x, &y);

        // Check if the mean squared error is computed correctly
        assert_eq!(mse, 9.0);
    }

    #[test]
    fn test_mean_squared_error_gradient_zeros() {
        // Create a simple series and dataframe for testing
        let x: DataFrame = DataFrame::new(vec![
            Series::new("x1", &[1.0, 2.0, 3.0]),
            Series::new("x2", &[1.0, 2.0, 3.0]),
        ])
        .unwrap();
        let y: Series = Series::new("y", &[1.0, 2.0, 3.0]);

        // Compute the mean squared error
        let gradient: Series = LossFunctionType::MeanSquaredError.gradient(&x, &y, &y);

        // Check if the mean squared error is computed correctly
        assert_eq!(gradient, Series::new("gradients", &[0.0, 0.0]));
    }

    #[test]
    fn test_mean_squared_error_gradient_non_zeros() {
        // Create a dataframe and two simple series for testing
        let x: DataFrame = DataFrame::new(vec![
            Series::new("x1", &[1.0, 2.0, 3.0]),
            Series::new("x2", &[4.0, 5.0, 6.0]),
        ])
        .unwrap();
        let y: Series = Series::new("y", &[1.0, 2.0, 3.0]);
        let y_pred: Series = Series::new("y_pred", &[4.0, 5.0, 6.0]);

        // Compute the mean squared error
        let gradient: Series = LossFunctionType::MeanSquaredError.gradient(&x, &y, &y_pred);

        // Check if the mean squared error is computed correctly
        assert_eq!(gradient, Series::new("gradients", &[12.0, 30.0]));
    }

    #[test]
    fn test_linear_model_fit_predict_single_feature() {
        // Sample data
        let x: Result<DataFrame, PolarsError> =
            DataFrame::new(vec![Series::new("feature1", vec![1, 2, 3])]);

        let y: Series = Series::new("target", vec![10.0, 20.0, 30.0]);

        // Create a linear model
        let mut model: Linear = Linear::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(1000, 0.1).is_ok());

        // Predict using the same data
        let predictions = model.predict(&x.unwrap()).unwrap();

        // Print out the values for debugging
        println!("Predictions: {:?}", predictions);
        println!("Actual values: {:?}", y);

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        // Print out the sums for debugging
        println!("Sum of predictions: {:?}", predictions.sum::<f64>());
        println!("Sum of actual values: {:?}", y.sum::<f64>());

        // Check the sums
        assert!(
            (predictions.sum::<f64>().unwrap() - y.sum::<f64>().unwrap()).abs() < 1e-6,
            "Sums do not match within epsilon"
        );

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        assert_eq!(predictions.sum::<f64>(), y.sum()); // A simple example
    }

    #[test]
    fn test_linear_model_fit_predict_multiple_features() {
        // Sample data
        let x: Result<DataFrame, PolarsError> = DataFrame::new(vec![
            Series::new("feature1", vec![1, 2, 3]),
            Series::new("feature2", vec![1, 2, 3]),
        ]);

        let y: Series = Series::new("target", vec![10.0, 20.0, 30.0]);

        // Create a linear model
        let mut model: Linear = Linear::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Predict using the same data
        let predictions = model.predict(&x.unwrap()).unwrap();

        // Print out the values for debugging
        println!("Predictions: {:?}", predictions);
        println!("Actual values: {:?}", y);

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        // Print out the sums for debugging
        println!("Sum of predictions: {:?}", predictions.sum::<f64>());
        println!("Sum of actual values: {:?}", y.sum::<f64>());

        // Check the sums
        assert!(
            (predictions.sum::<f64>().unwrap() - y.sum::<f64>().unwrap()).abs() < 1e-6,
            "Sums do not match within epsilon"
        );

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        assert_eq!(predictions.sum::<f64>(), y.sum()); // A simple example
    }

    #[test]
    fn test_linear_model_accuracy_perfect_single_feature() {
        // Sample data
        let x: Result<DataFrame, PolarsError> =
            DataFrame::new(vec![Series::new("feature1", vec![1, 2, 3])]);

        let y: Series = Series::new("target", vec![10.0, 20.0, 30.0]);

        // Create a linear model
        let mut model: Linear = Linear::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(1000, 0.1).is_ok());

        // Compute the accuracy
        let accuracy = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 1.0).abs() < 1e-6,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_linear_model_accuracy_non_perfect_single_feature() {
        // Sample data
        let x: Result<DataFrame, PolarsError> =
            DataFrame::new(vec![Series::new("feature1", vec![1, 2, 3, 4, 5])]);

        let y: Series = Series::new("target", vec![10.0, 25.0, 50.0, 56.0, 50.0]);

        // Create a linear model
        let mut model: Linear = Linear::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Compute the accuracy
        let accuracy = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 0.758).abs() < 1e-3,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_linear_model_accuracy_perfect_multiple_features() {
        // Sample data
        let x: Result<DataFrame, PolarsError> = DataFrame::new(vec![
            Series::new("feature1", vec![1, 2, 3]),
            Series::new("feature2", vec![1, 2, 3]),
        ]);

        let y: Series = Series::new("target", vec![10.0, 20.0, 30.0]);

        // Create a linear model
        let mut model: Linear = Linear::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Compute the accuracy
        let accuracy = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 1.0).abs() < 1e-6,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_linear_model_accuracy_non_perfect_multiple_features() {
        // Sample data
        let x: Result<DataFrame, PolarsError> = DataFrame::new(vec![
            Series::new("feature1", vec![1, 2, 3, 4, 5]),
            Series::new("feature2", vec![1, 2, 3, 4, 5]),
        ]);

        let y: Series = Series::new("target", vec![10.0, 25.0, 50.0, 56.0, 50.0]);

        // Create a linear model
        let mut model: Linear = Linear::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Compute the accuracy
        let accuracy = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 0.758).abs() < 1e-3,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_logistic_model_fit_predict_single_feature() {
        // Sample data
        let x: Result<DataFrame, PolarsError> =
            DataFrame::new(vec![Series::new("feature1", vec![1, 2, 3])]);

        let y: Series = Series::new("target", vec![0.0, 1.0, 1.0]);

        // Create a logistic model
        let mut model: Logistic = Logistic::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(1000, 0.1).is_ok());

        // Predict using the same data
        let predictions: Series = model.predict(&x.unwrap()).unwrap();

        // Round the predictions
        let predictions: Series = predictions
            .f64()
            .unwrap()
            .apply(|value: Option<f64>| {
                if value.is_nan() {
                    None
                } else {
                    Some(value.unwrap().round())
                }
            })
            .into_series();

        // Print out the values for debugging
        println!("Predictions: {:?}", predictions);
        println!("Actual values: {:?}", y);

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        // Print out the sums for debugging
        println!("Sum of predictions: {:?}", predictions.sum::<f64>());
        println!("Sum of actual values: {:?}", y.sum::<f64>());

        // Check the sums
        assert!(
            (predictions.sum::<f64>().unwrap() - y.sum::<f64>().unwrap()).abs() < 1e-6,
            "Sums do not match within epsilon"
        );

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        assert_eq!(predictions.sum::<f64>(), y.sum()); // A simple example
    }

    #[test]
    fn test_logistic_model_fit_predict_multiple_features() {
        // Sample data
        let x: Result<DataFrame, PolarsError> = DataFrame::new(vec![
            Series::new("feature1", vec![1, 3, 2]),
            Series::new("feature2", vec![1, 3, 2]),
        ]);

        let y: Series = Series::new("target", vec![1.0, 0.0, 1.0]);

        // Create a logistic model
        let mut model: Logistic = Logistic::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Predict using the same data
        let predictions: Series = model.predict(&x.unwrap()).unwrap();

        // Round the predictions
        let predictions: Series = Logistic::round_predictions(&predictions);

        // Print out the values for debugging
        println!("Predictions: {:?}", predictions);
        println!("Actual values: {:?}", y);

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        // Print out the sums for debugging
        println!("Sum of predictions: {:?}", predictions.sum::<f64>());
        println!("Sum of actual values: {:?}", y.sum::<f64>());

        // Check the sums
        assert!(
            (predictions.sum::<f64>().unwrap() - y.sum::<f64>().unwrap()).abs() < 1e-6,
            "Sums do not match within epsilon"
        );

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        assert_eq!(predictions.sum::<f64>(), y.sum()); // A simple example
    }

    #[test]
    fn test_logistic_model_accuracy_perfect_single_feature() {
        // Sample data
        let x: Result<DataFrame, PolarsError> =
            DataFrame::new(vec![Series::new("feature1", vec![1, 2, 3])]);

        let y: Series = Series::new("target", vec![0.0, 1.0, 1.0]);

        // Create a logistic model
        let mut model: Logistic = Logistic::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(1000, 0.1).is_ok());

        // Compute the accuracy
        let accuracy: f64 = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 1.0).abs() < 1e-6,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_logistic_model_accuracy_non_perfect_single_feature() {
        // Sample data
        let x: Result<DataFrame, PolarsError> =
            DataFrame::new(vec![Series::new("feature1", vec![1, 2, 3, 4, 5])]);

        let y: Series = Series::new("target", vec![0.0, 1.0, 1.0, 0.0, 1.0]);

        // Create a logistic model
        let mut model: Logistic = Logistic::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Compute the accuracy
        let accuracy: f64 = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 0.8).abs() < 1e-6,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_logistic_model_accuracy_perfect_multiple_features() {
        // Sample data
        let x: Result<DataFrame, PolarsError> = DataFrame::new(vec![
            Series::new("feature1", vec![1, 3, 2]),
            Series::new("feature2", vec![1, 3, 2]),
        ]);

        let y: Series = Series::new("target", vec![1.0, 0.0, 1.0]);

        // Create a logistic model
        let mut model: Logistic = Logistic::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Compute the accuracy
        let accuracy: f64 = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 1.0).abs() < 1e-6,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_logistic_model_accuracy_non_perfect_multiple_features() {
        // Sample data
        let x: Result<DataFrame, PolarsError> = DataFrame::new(vec![
            Series::new("feature1", vec![1, 3, 2, 4, 5]),
            Series::new("feature2", vec![1, 3, 2, 4, 5]),
        ]);

        let y: Series = Series::new("target", vec![1.0, 0.0, 1.0, 0.0, 1.0]);

        // Create a logistic model
        let mut model: Logistic = Logistic::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Compute the accuracy
        let accuracy: f64 = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 0.6).abs() < 1e-6,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_activation_function_sigmoid() {
        // Create a simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);

        // Compute the sigmoid
        let sigmoid: Series = ActivationFunctionType::Sigmoid.activate(&x);

        // Check if the sigmoid is computed correctly
        assert_eq!(
            sigmoid,
            Series::new(
                "activated_values",
                &[
                    1.0 / (1.0 + f64::exp(-1.0)),
                    1.0 / (1.0 + f64::exp(-2.0)),
                    1.0 / (1.0 + f64::exp(-3.0))
                ]
            )
        );
    }

    #[test]
    fn test_activation_function_identity() {
        // Create a simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);

        // Compute the identity
        let identity: Series = ActivationFunctionType::Identity.activate(&x);

        // Check if the identity is computed correctly
        assert_eq!(identity, Series::new("activated_values", &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_kernel_function_linear() {
        // Create a simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);

        // Compute the linear kernel
        let linear_kernel: Series = KernelFunctionType::Linear.kernel(&x, &x);

        // Check if the linear kernel is computed correctly
        assert_eq!(linear_kernel, Series::new("kernel", &[1.0, 4.0, 9.0]));
    }

    #[test]
    fn test_kernel_function_polynomial() {
        // Create a simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);

        // Compute the polynomial kernel
        let polynomial_kernel: Series = KernelFunctionType::Polynomial(2.0, 2.0).kernel(&x, &x);

        // Check if the polynomial kernel is computed correctly
        assert_eq!(
            polynomial_kernel,
            Series::new("kernel", &[9.0, 36.0, 121.0])
        );
    }

    #[test]
    fn test_kernel_function_rbf() {
        // Create two simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);
        let y: Series = Series::new("y", &[3.0, 2.0, 1.0]);

        // Compute the rbf kernel
        let rbf_kernel: Series = KernelFunctionType::RadialBasisFunction(2.0).kernel(&x, &y);

        // Check if the rbf kernel is computed correctly
        assert_eq!(
            rbf_kernel,
            Series::new("kernel", &[f64::exp(-8.0), f64::exp(0.0), f64::exp(-8.0)])
        );
    }

    #[test]
    fn test_hinge_loss_zero() {
        // Create a simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);

        // Compute the hinge loss
        let hinge_loss: f64 = LossFunctionType::Hinge.loss(&x, &x);

        // Check if the hinge loss is computed correctly
        assert_eq!(hinge_loss, 0.0);
    }

    #[test]
    fn test_hinge_loss_non_zero() {
        // Create two simple series for testing
        let y: Series = Series::new("y", &[1.0, 1.0, 1.0]);
        let y_pred: Series = Series::new("y_pred", &[1.0, 1.0, 0.0]);

        // Compute the hinge loss
        let hinge_loss: f64 = LossFunctionType::Hinge.loss(&y, &y_pred);

        // Check if the hinge loss is computed correctly
        assert_eq!(hinge_loss, 1.0);
    }

    #[test]
    fn test_svm_model_fit_predict_single_feature() {
        // Sample data
        let x: Result<DataFrame, PolarsError> =
            DataFrame::new(vec![Series::new("feature1", vec![1.0, 2.0, 3.0])]);

        let y: Series = Series::new("target", vec![0.0, 1.0, 1.0]);

        // Create a SVM model
        let mut model: SVM = SVM::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(1000, 0.1).is_ok());

        // Predict using the same data
        let predictions: Series = model.predict(&x.unwrap()).unwrap();

        // Round the predictions
        let predictions: Series = Logistic::round_predictions(&predictions);

        // Print out the values for debugging
        println!("Predictions: {:?}", predictions);
        println!("Actual values: {:?}", y);

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        // Print out the sums for debugging
        println!("Sum of predictions: {:?}", predictions.sum::<f64>());
        println!("Sum of actual values: {:?}", y.sum::<f64>());

        // Check the sums
        assert!(
            (predictions.sum::<f64>().unwrap() - y.sum::<f64>().unwrap()).abs() < 1e-6,
            "Sums do not match within epsilon"
        );

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        assert_eq!(predictions.sum::<f64>(), y.sum()); // A simple example
    }

    #[test]
    fn test_svm_model_fit_predict_multiple_features() {
        // Sample data
        let x: Result<DataFrame, PolarsError> = DataFrame::new(vec![
            Series::new("feature1", vec![1.0, 2.0, 3.0]),
            Series::new("feature2", vec![1.0, 2.0, 3.0]),
        ]);

        let y: Series = Series::new("target", vec![0.0, 1.0, 1.0]);

        // Create a SVM model
        let mut model: SVM = SVM::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Predict using the same data
        let predictions: Series = model.predict(&x.unwrap()).unwrap();

        // Round the predictions
        let predictions: Series = Logistic::round_predictions(&predictions);

        // Print out the values for debugging
        println!("Predictions: {:?}", predictions);
        println!("Actual values: {:?}", y);

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        // Print out the sums for debugging
        println!("Sum of predictions: {:?}", predictions.sum::<f64>());
        println!("Sum of actual values: {:?}", y.sum::<f64>());

        // Check the sums
        assert!(
            (predictions.sum::<f64>().unwrap() - y.sum::<f64>().unwrap()).abs() < 1e-6,
            "Sums do not match within epsilon"
        );

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        assert_eq!(predictions.sum::<f64>(), y.sum()); // A simple example
    }

    #[test]
    fn test_svm_model_accuracy_perfect_single_feature() {
        // Sample data
        let x: Result<DataFrame, PolarsError> =
            DataFrame::new(vec![Series::new("feature1", vec![1.0, 2.0, 3.0])]);

        let y: Series = Series::new("target", vec![0.0, 1.0, 1.0]);

        // Create a SVM model
        let mut model: SVM = SVM::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(1000, 0.1).is_ok());

        // Compute the accuracy
        let accuracy: f64 = model.accuracy(&x.as_ref().unwrap(), &y).unwrap();

        // Print out the actual and predicted values for debugging
        println!("Actual values: {:?}", y);
        println!(
            "Predicted values: {:?}",
            model.predict(&x.unwrap()).unwrap()
        );

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 1.0).abs() < 1e-6,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_svm_model_accuracy_non_perfect_single_feature() {
        // Sample data
        let x: Result<DataFrame, PolarsError> = DataFrame::new(vec![Series::new(
            "feature1",
            vec![1.0, -2.0, 3.0, 4.0, 5.0],
        )]);

        let y: Series = Series::new("target", vec![0.0, 0.0, 1.0, 0.0, 1.0]);

        // Create a SVM model
        let mut model: SVM = SVM::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Compute the accuracy
        let accuracy: f64 = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 0.8).abs() < 1e-6,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_svm_model_accuracy_perfect_multiple_features() {
        // Sample data
        let x: Result<DataFrame, PolarsError> = DataFrame::new(vec![
            Series::new("feature1", vec![1.0, 2.0, 3.0]),
            Series::new("feature2", vec![1.0, 2.0, 3.0]),
        ]);

        let y: Series = Series::new("target", vec![1.0, 1.0, 0.0]);

        // Create a SVM model
        let mut model: SVM = SVM::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Compute the accuracy
        let accuracy: f64 = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 1.0).abs() < 1e-6,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_svm_model_accuracy_non_perfect_multiple_features() {
        // Sample data
        let x: Result<DataFrame, PolarsError> = DataFrame::new(vec![
            Series::new("feature1", vec![1.0, -2.0, 3.0, 4.0, 5.0]),
            Series::new("feature2", vec![1.0, 2.0, 3.0, -4.0, 5.0]),
        ]);

        let y: Series = Series::new("target", vec![1.0, 1.0, 0.0, 0.0, 1.0]);

        // Create a SVM model
        let mut model: SVM = SVM::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(10000, 0.01).is_ok());

        // Compute the accuracy
        let accuracy: f64 = model.accuracy(&x.unwrap(), &y).unwrap();

        // Print out the accuracy for debugging
        println!("Accuracy: {:?}", accuracy);

        // Check the accuracy
        assert!(
            (accuracy - 0.4).abs() < 1e-6,
            "Accuracy does not match within epsilon"
        );
    }

    #[test]
    fn test_activation_function_relu() {
        // Create a simple series for testing
        let x: Series = Series::new("x", &[-1.0, 2.0, -3.0]);

        // Compute the relu
        let relu: Series = ActivationFunctionType::ReLU.activate(&x);

        // Check if the relu is computed correctly
        assert_eq!(relu, Series::new("activated_values", &[0.0, 2.0, 0.0]));
    }

    #[test]
    fn test_activation_function_identity_gradient() {
        // Create a simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);

        // Compute the identity gradient
        let identity_gradient: Series = ActivationFunctionType::Identity.gradient(&x);

        // Check if the identity gradient is computed correctly
        assert_eq!(
            identity_gradient,
            Series::new("gradients", &[1.0, 1.0, 1.0])
        );
    }

    #[test]
    fn test_activation_function_sigmoid_gradient() {
        fn sigmoid(x: f64) -> f64 {
            1.0 / (1.0 + f64::exp(-x))
        }

        // Create a simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);

        // Compute the sigmoid gradient
        let sigmoid_gradient: Series = ActivationFunctionType::Sigmoid.gradient(&x);

        // Check if the sigmoid gradient is computed correctly
        assert_eq!(
            sigmoid_gradient,
            Series::new(
                "gradients",
                &[
                    sigmoid(1.0) * (1.0 - sigmoid(1.0)),
                    sigmoid(2.0) * (1.0 - sigmoid(2.0)),
                    sigmoid(3.0) * (1.0 - sigmoid(3.0))
                ]
            )
        );
    }

    #[test]
    fn test_activation_function_relu_gradient() {
        // Create a simple series for testing
        let x: Series = Series::new("x", &[-1.0, 2.0, -3.0]);

        // Compute the relu gradient
        let relu_gradient: Series = ActivationFunctionType::ReLU.gradient(&x);

        // Check if the relu gradient is computed correctly
        assert_eq!(relu_gradient, Series::new("gradients", &[0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_linear_layer_new() {
        // Create a linear layer
        let linear_layer: LinearLayer = LinearLayer::new(Series::new("weights", [1.0, 2.0, 3.0]),Series::new("biases", [1.0, 2.0, 3.0])); 

        // Check if the linear layer is created correctly
        assert_eq!(linear_layer.weights, Series::new("weights", vec![1.0, 2.0, 3.0]));
        assert_eq!(linear_layer.biases, Series::new("biases", vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_linear_layer_zeroes() {
        // Create a linear layer
        let linear_layer: LinearLayer = LinearLayer::zeroes(3);

        // Check if the linear layer is created correctly
        assert_eq!(linear_layer.weights, Series::new("weights", vec![0.0, 0.0, 0.0]));
        assert_eq!(linear_layer.biases, Series::new("biases", vec![0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_linear_layer_new_random() {
        // Create a linear layer
        let linear_layer: LinearLayer = LinearLayer::new_random(5, [-1.0, 1.0]);

        // Check if the linear layer is created correctly
        assert_eq!(linear_layer.weights.len(), 5);
        assert_eq!(linear_layer.biases.len(), 5);

        // Check if the weights and biases are within the specified range
        assert!(linear_layer.weights.f64().unwrap().min().unwrap() >= -1.0);
        assert!(linear_layer.weights.f64().unwrap().max().unwrap() <= 1.0);
    }

    #[test]
    fn test_linear_layer_forward() {
        // parameters
        let weights: Series = Series::new("weights", vec![1.0, 2.0, 3.0]);
        let biases: Series = Series::new("biases", vec![4.0, 5.0, 6.0]);
        let activation_function: ActivationFunctionType = ActivationFunctionType::Sigmoid;
        let input: Series = Series::new("input", vec![7.0, 8.0, 9.0]);

        let sigmoid = |x: f64| -> f64 { return 1.0 / (1.0 + f64::exp(-x)) };

        // Create a linear layer
        let linear_layer: LinearLayer = LinearLayer::new(weights.clone(), biases.clone());

        // Compute the forward pass
        let output: Series = linear_layer.forward(input.clone(), activation_function.clone());

        assert_eq!(
            output,
            Series::new(
                "activated_values",
                &[
                    sigmoid((&input * &weights).sum::<f64>().unwrap() + &biases.f64().unwrap().get(0).unwrap()),
                    sigmoid((&input * &weights).sum::<f64>().unwrap() + &biases.f64().unwrap().get(1).unwrap()),
                    sigmoid((&input * &weights).sum::<f64>().unwrap() + &biases.f64().unwrap().get(2).unwrap())
                ]
            )
        );
    }

    // #[test]
    // fn test_linear_layer_backward() {
    //     // parameters
    //     let weights: Series = Series::new("weights", vec![1.0, 2.0, 3.0]);
    //     let biases: Series = Series::new("biases", vec![4.0, 5.0, 6.0]);
    //     let activation_function: ActivationFunctionType = ActivationFunctionType::ReLU;
    //     let loss_function: LossFunctionType = LossFunctionType::Hinge;
    //     let input: Series = Series::new("input", vec![7.0, 8.0, 9.0]);

    //     // Create a linear layer
    //     let test_layer: LinearLayer = LinearLayer::new(weights.clone(), biases.clone());

    //     // Compute the forward pass

    //     let output: Series = test_layer.forward(input.clone(), activation_function.clone());

    //     // Compute the backward pass
    //     let (d_input, d_weights, d_biases): (Series, Series, Series) = test_layer.backward(
    //         input.clone(),
    //         output.clone(),
    //         loss_function.clone,
    //         activation_function,
    //         loss_function.gradient(&input.clone().into_frame(), &output.clone(), &output.clone()),
    //     );
    // }
}
