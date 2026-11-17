# Cloud-Based AI Digital Twin for Business Risk Prediction and Profit Optimization

A B.Tech AI & Data Science / Introduction to Cloud Computing mini-project combining demand forecasting, business what-if simulation, cloud resource simulation, security concepts, and SLA monitoring.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL shown in the terminal.

## Demo login

- `admin` / `ChangeMe123!`
- `manager` / `Manager123!`

These credentials are only for demonstration. Do not deploy this login setup for real users.

## Optional CSV input

The app accepts a CSV with at least these columns:

- `date` (date)
- `units_sold` (numeric)

Optional columns:

- `price` (numeric; defaults to 20)
- `promotion` (0/1; defaults to 0)

At least 30 valid daily rows are recommended. Built-in data is synthetic.

## Modules

1. Business Overview
2. AI Demand Forecast
3. What-if Business Digital Twin
4. Cloud Architecture & Resource Management Simulator
5. Security & SLA
6. About, limitations and ICC syllabus mapping

## ICC syllabus mapping

- Unit 1: Distributed-computing concepts explained in the report.
- Unit 2: Virtual server simulation; virtualization and hypervisors explained.
- Unit 3: Cloud basics, SaaS/PaaS/IaaS, deployment discussion.
- Unit 4: Resource pooling, elasticity, scalability and load-balancing simulation.
- Unit 5: Load-balanced instances and resource management simulation; remaining advanced patterns explained in report.
- Unit 6: Demo authentication, identity/access-control concepts and security limitations.
- Unit 7: SLA example, illustrative billing estimate, and cloud provider case study in report.

## Limitations

The cloud architecture controls are simulations, not actual virtual-machine provisioning or real auto-scaling. Forecasts are educational and are not financial or operational guarantees. The built-in data is synthetic.

## Author

Hensi Solanki, B.Tech Artificial Intelligence & Data Science, A. D. Patel Institute of Technology (CVM University)
