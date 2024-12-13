# NHSN Streamlit App

A Python app to display up-to-date NHSN data, including:
- Fraction Reporting
- Weekly Admission Data

---

## Setting Up the Environment

To run this app, you need to create a Python environment using **conda** or **venv**.

### Required Python Packages:
- `streamlit`
- `numpy`
- `matplotlib`
- `pandas`
- `datetime`
- `epiweeks`

1. **Create and Activate the Environment**:
   - Using `conda`:
     ```bash
     conda create -n nhsn-env python=3.9
     conda activate nhsn-env
     ```
   - Using `venv`:
     ```bash
     python -m venv nhsn-env
     source nhsn-env/bin/activate  # On Windows: nhsn-env\Scripts\activate
     ```

2. **Install Required Packages**:
   ```bash
   pip install streamlit numpy matplotlib pandas datetime epiweeks
    ```
   
3. **Clone the Repository**:
   ```git clone <repository_url>
   cd nhsn-streamlit
   ```

## Running the App

   ```
   streamlit run nhsn.app
  ```
## Support

For Questions/Comments please email: mbennun@predsci.com
