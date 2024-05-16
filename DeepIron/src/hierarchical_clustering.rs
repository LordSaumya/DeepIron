//! A set of structures and functions for hierarchical clustering.

use crate::data_loader::DataFrameTransformer;
use crate::model::activation_functions::{ActivationFunction, ActivationFunctionType};
use crate::model::loss_functions::{LossFunction, LossFunctionType};
use crate::model::*;
use polars::prelude::*;
use polars::series::Series;

/// A struct that defines a hierarchical clustering model.
/// 
/// # Example
/// 
/// ```
/// 
/// let model = Model::HierarchicalClustering::new(4, DistMeasure::SLink);
/// 
/// model.fit(&x);
/// 
/// let y = model.predict(&x);
/// 
/// ```
pub struct HierarchicalClustering {
    /// The number of clusters to be formed.
    pub n_clusters: usize,
    /// The distance measure to be used in hierarchical clustering.
    pub dist_measure: DistMeasure,
    /// The clusters formed by the model, stored as a vector of DataFrames with each DataFrame containing the data points of a cluster in each column.
    pub clusters: Vec<DataFrame>,
}

/// A struct that defines the distance measure to be used in hierarchical clustering.
/// 
/// # Example
/// 
/// ```
/// 
/// let dist_measure = DistMeasure::SLink;
/// 
/// ```
#[derive(Clone)]
pub enum DistMeasure {
    /// Single Linkage
    /// The minimum distance between elements of two clusters.
    SLink,
    /// Complete Linkage
    /// The maximum distance between elements of two clusters.
    CLink,
    /// Centroid Linkage
    /// The distance between the centroids of two clusters.
    Centroid,
}

impl HierarchicalClustering {
    /// Create a new hierarchical clustering model with the specified number of clusters and distance measure.
    /// 
    /// # Arguments
    /// 
    /// * `n_clusters` - The number of clusters to be formed.
    /// 
    /// * `dist_measure` - The distance measure to be used in hierarchical clustering.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let model = Model::HierarchicalClustering::new(4, DistMeasure::SLink);
    /// 
    /// ```
    pub fn new(n_clusters: usize, dist_measure: DistMeasure) -> HierarchicalClustering {
        HierarchicalClustering {
            n_clusters,
            dist_measure,
            clusters: Vec::new(),
        }
    }

    /// Calculate the Euclidean distance between two points.
    /// 
    /// # Arguments
    /// 
    /// * `point1` - A Series containing the coordinates of the first point.
    /// 
    /// * `point2` - A Series containing the coordinates of the second point.
    /// 
    /// # Returns
    /// 
    /// * `f64` - The Euclidean distance between the two points.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let distance = Model::KMeans::euclidean_distance(&point1, &point2);
    /// 
    /// ```
    fn euclidean_distance(point1: &Series, point2: &Series) -> f64 {
        let point1: Vec<f64> = point1.f64().unwrap().into_iter().map(|x| x.unwrap()).collect();
        let point2: Vec<f64> = point2.f64().unwrap().into_iter().map(|x| x.unwrap()).collect();

        let mut sum: f64 = 0.0;

        for (i, _) in point1.iter().enumerate() {
            sum += (point1[i] - point2[i]).powi(2);
        }
        sum.sqrt()
    }

    /// Calculate the distance between two clusters.
    /// 
    /// # Arguments
    /// 
    /// * `cluster1` - A DataFrame containing the data points of the first cluster.
    /// 
    /// * `cluster2` - A DataFrame containing the data points of the second cluster.
    /// 
    /// * `dist_measure` - The distance measure to be used in hierarchical clustering.
    /// 
    /// # Returns
    /// 
    /// * `f64` - The distance between the two clusters.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let distance = Model::KMeans::cluster_distance(&cluster1, &cluster2, DistMeasure::SLink);
    /// 
    /// ```
    fn cluster_distance(cluster1: &DataFrame, cluster2: &DataFrame, dist_measure: DistMeasure) -> f64 {
        let mut distances = Vec::new();
        for i in 0..cluster1.width() {
            for j in 0..cluster2.width() {
                let distance = HierarchicalClustering::euclidean_distance(&cluster1.column(i).unwrap(), &cluster2.column(j).unwrap());
                distances.push(distance);
            }
        }
        match dist_measure {
            DistMeasure::SLink => *distances.iter().min().unwrap(),
            DistMeasure::CLink => *distances.iter().max().unwrap(),
            DistMeasure::Centroid => {
                let mut centroid1 = Vec::new();
                let mut centroid2 = Vec::new();
                for i in 0..cluster1.width() {
                    // Each row in the cluster is one feature of all the data points
                    let sum1: f64 = cluster1.get(i).unwrap().f64().unwrap().into_iter().map(|x| x.unwrap()).sum();
                    centroid1.push(sum1 / cluster1.height() as f64);
                    let sum2: f64 = cluster2.get(i).unwrap().f64().unwrap().into_iter().map(|x| x.unwrap()).sum();
                    centroid2.push(sum2 / cluster2.height() as f64);
                }
                HierarchicalClustering::euclidean_distance(&Series::new("centroid1", &centroid1), &Series::new("centroid2", &centroid2))
            }
        }
    }
}

impl model::ClusterModeller for HierarchicalClustering {
    /// Fit the hierarchical clustering model to the input data.
    /// 
    /// # Arguments
    /// 
    /// * `x` - The input data.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// model.fit(&x);
    /// 
    /// ```
    fn fit(&mut self, x: &DataFrame) {
        // Initialise the clusters with each data point as a cluster
        for i in 0..x.height() {
            let mut cluster = DataFrame::new(x.width(), x.get_column_names());
            for j in 0..x.width() {
                cluster.add_series(x.column(j).unwrap());
            }
            self.clusters.push(cluster);
        }

        // Merge clusters until the number of clusters is equal to the specified number of clusters
        while self.clusters.len() > self.n_clusters {
            let mut min_distance = f64::INFINITY;
            let mut cluster1_index = 0;
            let mut cluster2_index = 0;

            for i in 0..self.clusters.len() {
                for j in i + 1..self.clusters.len() {
                    let distance = HierarchicalClustering::cluster_distance(&self.clusters[i], &self.clusters[j], self.dist_measure.clone());
                    if distance < min_distance {
                        min_distance = distance;
                        cluster1_index = i;
                        cluster2_index = j;
                    }
                }
            }

            let cluster1 = self.clusters.remove(cluster1_index);
            let cluster2 = self.clusters.remove(cluster2_index);

            let mut merged_cluster = DataFrame::new(x.width(), x.get_column_names());
            for i in 0..x.width() {
                for j in 0..cluster1.height() {
                    merged_cluster.add_series(cluster1.column(i).unwrap());
                }
                for j in 0..cluster2.height() {
                    merged_cluster.add_series(cluster2.column(i).unwrap());
                }
            }

            self.clusters.push(merged_cluster);
        }
    }

    /// Predict the cluster of each data point in the input data.
    /// 
    /// # WARNING: Since hierarchical clustering is not a predictive model, this function discards the input data and returns the clusters formed by the model as a Series of DataFrames.
    /// 
    /// # Arguments
    /// 
    /// * `x` - The input data.
    /// 
    /// # Returns
    /// 
    /// * `Series` - A Series containing the clusters formed by the model on the input data.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let y = model.predict(&x);
    /// 
    /// ```
    fn predict(&mut self, x: &DataFrame) -> Result<Series, PolarsError> {
        let mut cluster_series: Series = Series::new("clusters", &self.clusters);
        Ok(cluster_series)
    }

    /// Calculate the compactness of the clusters formed by the model using sum of squared errors.
    /// 
    /// # Arguments
    /// 
    /// * `x` - The input data.
    /// 
    /// # Returns
    /// 
    /// * `f64` - The compactness of the clusters formed by the model.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let compactness = model.compactness(&x);
    /// 
    /// ```
    fn compactness(&mut self, x: &DataFrame) -> Result<f64, PolarsError> {
        let mut compactness = 0.0;
        for i in 0..self.clusters.len() {
            for j in 0..self.clusters[i].height() {
                compactness += HierarchicalClustering::euclidean_distance(&x.row(j), &self.clusters[i].row(j)).powi(2);
            }
        }
        Ok(compactness)
    }
}
