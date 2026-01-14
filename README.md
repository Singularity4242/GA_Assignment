# Vehicle Routing Problem (VRP) Solver Using Genetic Algorithms
使用遗传算法解决车辆路径规划VRP问题
## Project Overview

This project implements a **Genetic Algorithm (GA)** to solve five variants of the Vehicle Routing Problem (VRP), a classic optimization challenge in logistics and supply chain management. The solver is designed to handle:

- **Classical VRP**  
- **Stochastic Demand VRP**  
- **Large-Scale VRP with Clustering**  
- **Multi-Objective VRP**  
- **Pickup and Delivery VRP**

The algorithm optimizes routes under constraints such as vehicle capacity, depot assignments, and demand uncertainty, aiming to minimize total travel distance and maximize efficiency.

---

## Project Structure

```
1_CLASSICAL_VRP.py
2_STOCHASTIC_DEMAND.py
3_LARGE_SCALE_OPTIMAZATION.py
4_MULTI-OBJECTIVE_OPTIMAZATION.py
5_PICKUP_DELIVERY.py
data_process.py                  # Data loading and preprocessing
visualizer.py                 # Route plotting and convergence graphs
VRP.csv                          # Dataset (customer and depot locations)
```

---

## Features Description

### **1_CLASSICAL_VRP.py**

Problem Type: Classic VRP

Objective: Optimal route for a vehicle with a capacity of 200, starting from depot 0, serving 100 customers, and returning to the starting point.

Optimization Objective: Minimize the total travel distance.

Characteristics: Defined requirements, single objective, single vehicle, single warehouse.

### **2_STOCHASTIC_DEMAND.py**

**Problem Type:** Stochastic Demand VRP

**Objective:** Considering the uncertainty of customer demand, the actual demand of each customer follows a normal distribution with mean DEMAND and standard deviation of 0.2×DEMAND, taking positive integer values.

**Optimization Objective:** Minimize the total travel distance under the condition of stochastic demand.

**Characteristics:** Stochastic demand, robust optimization, applicable to real-world logistics scenarios.

### **3_LARGE_SCALE_OPTIMAZATION.py**

**Problem Type**: Large-Scale VRP (with Clustering)

**Objective:** Increase the Y-coordinates of the original 100 customers by 150 to generate a large-scale dataset of 200 customers. Group the customers using clustering algorithms (K-means, DBSCAN). Vehicles must serve all customers in the same area before moving to the next area.

**Optimization Objective:** Minimize the total travel distance.

**Characteristics:** Large-scale problem, clustering preprocessing, regionalized path planning.

### **4_MULTI-OBJECTIVE_OPTIMAZATION.py**

**Problem Type:** Multi-objective optimization VRP

**Objective:** Simultaneously optimize total distance and total efficiency.

**Supports two methods:**

- Weighted method: Converts multi-objective optimization into single-objective optimization, with the objective function as follows: *f*=*w*×*f*1−(1−*w*)×*f*2

- Pareto method: Uses the NSGA-II algorithm to obtain the Pareto front solution set.

Users can switch optimization modes via input for easy comparison of the two methods.

### **5_PICKUP_DELIVERY.py**

**Problem Type:** Hybrid Pickup and Delivery VRP

**Objective:** Randomly select 30% of existing customers as pickup customers (negative demand), and the remainder as delivery customers (positive demand). The vehicle's net load must be between 0 and 200 at all times.

**Optimization Objective:** Minimize the total travel distance while meeting load constraints.

**Characteristics:** Hybrid demand, load constraints, suitable for two-way logistics scenarios.

---

## Installation & Setup

### Prerequisites
- Python 3.7+
- Required libraries: `numpy`, `pandas`, `matplotlib`, `deap`, `scikit_learn`

### Install Dependencies
```bash
pip install numpy pandas matplotlib deap scikit_learn
```

---

##  **Visualization**

**1_CLASSICAL_VRP.py**

![image-20260114170234659](https://github.com/Singularity4242/GA_Assignment/blob/main/imgs/image-20260114170234659.png)

![image-20260114170257060](https://github.com/Singularity4242/GA_Assignment/blob/main/imgs/image-20260114170257060.png)

**2_STOCHASTIC_DEMAND.py**

![image-20260114170315903](https://github.com/Singularity4242/GA_Assignment/blob/main/imgs/image-20260114170315903.png)

![image-20260114170323068](https://github.com/Singularity4242/GA_Assignment/blob/main/imgs/image-20260114170315903.png)

**3_LARGE_SCALE_OPTIMAZATION.py**

![image-20260114170405879](https://github.com/Singularity4242/GA_Assignment/blob/main/imgs/image-20260114170405879.png)

![image-20260114170412015](https://github.com/Singularity4242/GA_Assignment/blob/main/imgs/image-20260114170412015.png)

**4_MULTI-OBJECTIVE_OPTIMAZATION.py**

**5_PICKUP_DELIVERY.py**

![image-20260114170533104](https://github.com/Singularity4242/GA_Assignment/blob/main/imgs/image-20260114170533104.png)

![image-20260114170540144](https://github.com/Singularity4242/GA_Assignment/blob/main/imgs/image-20260114170540144.png)
