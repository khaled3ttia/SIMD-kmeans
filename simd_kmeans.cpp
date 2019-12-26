// Implementation of the KMeans Algorithm
// reference: http://mnemstudio.org/clustering-k-means-example-1.htm

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <x86intrin.h>


using namespace std;

class Point
{
private:
	int id_point, id_cluster;
	int total_values;
	string name;

public:
	vector<float> values;
	Point(int id_point, vector<float>& values, string name = "")
	{
		this->id_point = id_point;
		total_values = values.size();

		for(int i = 0; i < total_values; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

	int getID() const
	{
		return id_point;
	}

	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

	int getCluster() const
	{
		return id_cluster;
	}

	float getValue(int index) const 
	{
		return values[index];
	}

	int getTotalValues() const
	{
		return total_values;
	}

	void addValue(float value)
	{
		values.push_back(value);
	}

	string getName() const
	{
		return name;
	}
};

class Cluster
{
private:
	int id_cluster;
	

public:
	vector<float> central_values;
	vector<Point> points;
	Cluster(int id_cluster, const Point& point)
	{
		this->id_cluster = id_cluster;

		int total_values = point.getTotalValues();

		for(int i = 0; i < total_values; i++)
			central_values.push_back(point.getValue(i));

		points.push_back(point);
	}

	void addPoint(const Point& point)
	{
		points.push_back(point);
	}

	bool removePoint(int id_point)
	{
		int total_points = points.size();
		
		for(int i = 0; i < total_points; i++)
		{
			if(points[i].getID() == id_point)
			{
				points.erase(points.begin() + i);
				return true;
			}
		}
		
		return false;
	}

	float getCentralValue(int index) const
	{
		return central_values[index];
	}

	void setCentralValue(int index, float value)
	{
		central_values[index] = value;
	}

	Point getPoint(int index) const
	{
		return points[index];
	}

	int getTotalPoints() const
	{
		return points.size();
	}

	int getID()
	{
		return id_cluster;
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

	// return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(const Point& point)
	{
		float sum = 0.0, min_dist;
		int id_cluster_center = 0;
		
		//use x86 intrinstics here
		
		__m256 distance2, partialSum;
		partialSum = _mm256_setzero_ps();
		float dist[K];
		for(int i = 0; i < K; i++)
		{
			sum = 0.0;
			for (int j = 0; j < total_values; j += 8){
				__m256 pValues = _mm256_loadu_ps(&point.values[j]);
				__m256 centerValue = _mm256_loadu_ps(&clusters[i].central_values[j]);
				__m256 distance = _mm256_sub_ps(centerValue, pValues);
				__m256 distance1 = distance;
				distance2 = _mm256_mul_ps(distance, distance1);
				partialSum = _mm256_add_ps(distance2, partialSum);
			}
			distance2 = _mm256_setzero_ps();
			partialSum = _mm256_hadd_ps(partialSum, distance2);
			partialSum = _mm256_hadd_ps(partialSum, distance2);

			_mm256_store_ps(&sum, partialSum);

			dist[i] = sqrt(sum);
		}
		
		
		min_dist = dist[0];
		for (int i = 1 ; i < K; i ++){
			if (dist[i] < min_dist){
				min_dist = dist[i];
				id_cluster_center = i;
			}
						
		}

		return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	void run(vector<Point> & points)
	{
        auto begin = chrono::high_resolution_clock::now();
        
		if(K > total_points)
			return;

		vector<int> prohibited_indexes;

		// choose K distinct values for the centers of the clusters
		for(int i = 0; i < K; i++)
		{
			while(true)
			{
				int index_point = rand() % total_points;

				if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
						index_point) == prohibited_indexes.end())
				{
					prohibited_indexes.push_back(index_point);
					points[index_point].setCluster(i);
					Cluster cluster(i, points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
		}
        auto end_phase1 = chrono::high_resolution_clock::now();
        
		int iter = 1;

		while(true)
		{
			bool done = true;

			// associates each point to the nearest center
			for(int i = 0; i < total_points; i++)
			{
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);

				if(id_old_cluster != id_nearest_center)
				{
					if(id_old_cluster != -1)
						clusters[id_old_cluster].removePoint(points[i].getID());

					points[i].setCluster(id_nearest_center);
					clusters[id_nearest_center].addPoint(points[i]);
					done = false;
				}
			}
			
			// recalculating the center of each cluster
			
			__m256 partialSum2 = _mm256_setzero_ps();
			for(int i = 0; i < K; i++)
			{
				for(int j = 0; j < total_values; j += 8)
				{
					int total_points_cluster = clusters[i].getTotalPoints();
					float sum = 0.0;

					if(total_points_cluster > 0)
					{
						for(int p = 0; p < total_points_cluster-1; p += 2)
						{
							__m256 pValues = _mm256_loadu_ps(&clusters[i].points[p].values[j]);
							
							__m256 pValues2 = _mm256_loadu_ps(&clusters[i].points[p+1].values[j]);

							__m256 partialSum = _mm256_add_ps(pValues, pValues2);

							partialSum2 = _mm256_add_ps(partialSum, partialSum2);

						}
							
						_mm256_store_ps(&sum, partialSum2);
						clusters[i].setCentralValue(j, sum / total_points_cluster);
					}
				}
			}

			if(done == true || iter >= max_iterations)
			{
				cout << "Break in iteration " << iter << "\n\n";
				break;
			}

			iter++;
		}
        auto end = chrono::high_resolution_clock::now();

		// shows elements of clusters
		for(int i = 0; i < K; i++)
		{
			int total_points_cluster =  clusters[i].getTotalPoints();

			cout << "Cluster " << clusters[i].getID() + 1 << endl;
			for(int j = 0; j < total_points_cluster; j++)
			{
				cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
				for(int p = 0; p < total_values; p++)
					cout << clusters[i].getPoint(j).getValue(p) << " ";

				string point_name = clusters[i].getPoint(j).getName();

				if(point_name != "")
					cout << "- " << point_name;

				cout << endl;
			}

			cout << "Cluster values: ";

			for(int j = 0; j < total_values; j++)
				cout << clusters[i].getCentralValue(j) << " ";

			cout << "\n\n";
            cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";
            
            cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";
            
            cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
		}
	}
};

int main(int argc, char *argv[])
{
	srand (time(NULL));

	int total_points, total_values, K, max_iterations, has_name;

	cin >> total_points >> total_values >> K >> max_iterations >> has_name;

	vector<Point> points;
	string point_name;

	for(int i = 0; i < total_points; i++)
	{
		vector<float> values;

		for(int j = 0; j < total_values; j++)
		{
			float value;
			cin >> value;
			values.push_back(value);
		}

		if(has_name)
		{
			cin >> point_name;
			Point p(i, values, point_name);
			points.push_back(p);
		}
		else
		{
			Point p(i, values);
			points.push_back(p);
		}
	}

	KMeans kmeans(K, total_points, total_values, max_iterations);
	kmeans.run(points);

	return 0;
}
