#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Classify an original DTI value into one of the six analysis groups.
def find_dti_group(dti):

    # Estimate whether a value is a special value
    if dti in ["","NA","Exempt"]:
        group=None

    # Keep the interval labels already provided in the original data.
    elif dti=="<20%":
        group="<20%"
    elif dti =="20%-<30%":
        group="20%-<30%"
    elif dti=="30%-<36%":
        group="30%-<36%"
    elif dti=="50%-60%":
        group="50%-60%"
    elif dti==">60%":
        group=">60%"

    else:

        # Combine values from 36 to 49 into the same analysis group.
        dti_number=int(dti) 
        if dti_number>=36 and dti_number<50:
            group="36%-<50%"
        else:
            group=None
    return group

# Store results for one DTI group within a state and loan type.
class StateData:
    def __init__(self, state, loan_type, dti_bin, approval_rates, size):
        self.state = state  
        self.loan_type = loan_type 
        self.dti_bin = dti_bin 

        # Approval rate is a proportion between 0 and 1,
        # or None when the group has no valid applications.
        self.approval_rates = approval_rates

        # Number of valid applications used as the approval-rate denominator.
        self.size = size

    # Return a text summary of the stored result.
    def print_data(self):
        return f"state is {self.state}, Loan Type is {self.loan_type}, dti_bin is, {self.dti_bin}, Approval Rates is {self.approval_rates}, Size is {self.size}"

# A function that saves analyze result from singleStateAnalysis so that user don't take another minute to load the raw data
def data_saver(state, loan_type, dti_bins, approval_rates, size):

    # Store each group's label, approval rate and valid application count.
    for i in range(len(dti_bins)):
        new_data = StateData(state, loan_type, dti_bins[i], approval_rates[i], size[i])

        # Generate a dictionary key from the state, loan tyoe and DTI group.       
        new_data_key = (state, loan_type, dti_bins[i])

        # Handling dupliate entry
        if (new_data_key in saved_datasets_v2):
            overideYN = input(f"You are trying to save an existing record: {new_data.print_data()}, you sure want to overide? Y/N")

            if (overideYN == "Y" or overideYN == "y"):
                saved_datasets_v2[new_data_key] = new_data
        else:
            saved_datasets_v2[new_data_key] = new_data

# Load data from saved cache for a form suitable for analyzing
# Return an array with columns: DTI group,approval rate, valid application count.
def list_object_to_array_and_filter(list_state_data, state, loan_type):
    array = []
    for state_data in list_state_data:

        # Determine whether the current data are consistent with the specified data
        if (state_data.state == state and state_data.loan_type == loan_type):
            array.append([state_data.dti_bin, state_data.approval_rates, state_data.size])

    return np.array(array, dtype=object)

# Calculate valid application counts, approval counts and rates by DTI group.
# Each input record contains: DTI, loan type, action taken.
# Action codes are stored as strings.
def calculate_approval_rates(records):

    # dti bins
    groups=["<20%",
           "20%-<30%",
           "30%-<36%",
           "36%-<50%",
           "50%-60%",
           ">60%"]

    # Initializing
    totals=[0,0,0,0,0,0]
    approved_counts=[0,0,0,0,0,0]

    for record in records:
        dti=record[0]
        action=record[2]
        group=find_dti_group(dti)

       # Count the valid loan applications with operation codes 1, 2 or 3
        if action in ["1","2","3"]:

            # Count the application only if its DTI matches
            for i in range(len(groups)):
                if group==groups[i]:
                    totals[i]+=1

                    # Action codes 1 and 2 count as approved applications.
                    # 1: loan approved
                    # 2: Loan approved but not taken
                    if action in ["1","2"]:
                        approved_counts[i]+=1

    rates=[]

    # Calculate each rate as approved applications divided by valid applications. 
    for i in range(len(groups)):
        if totals[i] > 0:
            rate = approved_counts[i]/totals[i]
        else:
            # Set the approval rate to "None" to avoid dividing by 0
            rate = None

        rates.append(rate)

    # Return four lists in the same DTI group order.
    # Rates are proportions between 0 and 1, or None for empty groups.
    return groups, totals, approved_counts, rates

def filter(dataset, state, loan_type):
    # List the states available in the loaded data.
    city = dataset[:,0]
    loan = dataset[:,1]

    # create a Ture/False condition for the selected state.
    state_condition=city==state

    # Create a separate condition for the selected loan type.
    loan_type_condition=loan==loan_type

    # State and loan type filter.
    combined=state_condition&loan_type_condition
    filtered_data=dataset[combined]
    return filtered_data

# modular data extraction
def extract_data(state, loan_type, max_rows):

    path = "2025_combined_mlar_header.txt"

    # Columns of interest
    selected_columns = [
        "state_code",
        "loan_type",
        "action_taken",
        "debt_to_income_ratio"
    ]

    # Find the index of columns of interest
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip().split("|")

    column_indices = [
        header.index(col) for col in selected_columns
    ]

    # Initializing
    extracted_data_v2 = np.empty((0, 4), dtype=str)
    blocks = []
    f = open(path, "r", encoding="utf-8-sig")

    # Wrap this part into an exception capture, so that f.close() can always reached
    try:
        f.readline()

        # Load a block of raw data, filter them, attach it to other data until there's no new data left
        # This step will cause a warning about loading an empty dataset, it's because it's reaching the end of the data source, which is normal, expected and intended.
        while True:

            batch = np.genfromtxt(
                f,
                delimiter="|",
                usecols=column_indices,
                max_rows= max_rows,
                dtype=str
            )

            if batch.size == 0:
                break

            block = filter(batch, state, loan_type)
            if (block.size > 0):
                blocks.append(block)

        extracted_data_v2 = np.concatenate(blocks, axis=0)        
    except:
        print("something goes wrong while loading data")

    f.close()
    return extracted_data_v2

def check_input_valid(user_input, acceptable_list):
    user_input = user_input.upper()

    # if input "/cancel" if will return an empty string, which will be cheked in implementation and used to return to main 
    if user_input == "/CANCEL":
        return ""

    # A recursive design to recheck the input if the formerly entered value is not valid
    if user_input not in acceptable_list:
        print("you are inputing an invalid argument, please try again")
        new_user_input = input("enter a new argument:")
        return check_input_valid(new_user_input, acceptable_list)

    # Pass the input down if it's a valid input
    if user_input in acceptable_list:
        return user_input

# Analyse and plot DTI approval rates for a selected state and loan type.
# Users can optionally save the results for later comparison.
def analyze_state_DTI_approval_rate(max_rows):
    selected_columns = [
        "state_code",
        "loan_type",
        "action_taken",
        "debt_to_income_ratio"
    ]

    # Enter the arguments required to run the analysis, if input is an empty string, then return to main
    state_input = input("Enter your state as 2 capital letters, type /cancel if you wish to return to main:").strip().upper()
    state = check_input_valid(state_input, acceptable_states)
    if state == "":
        return

    loan_type_input=input("Enter your loan type as a integer number from 1 to 4, type /cancel if you wish to return to main, where you could check the meaning of loan types").strip()
    loan_type = check_input_valid(loan_type_input, acceptable_loan_type)
    if loan_type == "":
        return

    print("Extracting data, this may take roughly 1 minute")    
    extracted_data = extract_data(state, loan_type, max_rows)

    # Count missing values in each selected column for checking.
    missing_mask = np.isin(
        np.char.strip(extracted_data),
        ["", "NA", "nan", "NaN"]
    )
    print("=======================Data Quality Brief=======================")
    for i, name in enumerate(selected_columns):
        print(name, "missing:", np.sum(missing_mask[:, i]))
    # Reorder columns to : DTI, state, loan tyoe, action taken.
    extracted_data = extracted_data[:, [3, 0, 1, 2]]

    # Detect missing DTI and state values without changing the original data.
    dti_missing = np.isin(
        np.char.upper(np.char.strip(extracted_data[:, 0].astype(str))),
        ["", "NA", "NAN", "NONE", "NULL"]
    )

    state_missing = np.isin(
        np.char.upper(np.char.strip(extracted_data[:, 1].astype(str))),
        ["", "NA", "NAN", "NONE", "NULL"]
    )

    # Exclude rows with missing DTI or state information.
    # Exampt DTI values are handled later by find_dti_group().
    valid_rows = ~(state_missing | dti_missing)
    extracted_data = extracted_data[valid_rows]

    # Remove the state column before passing records to the calculation function.
    # The remaining columns are: DTI, loantype, action taken.
    extracted_data = np.delete(extracted_data, 1, axis=1)

    # Calculate counts and approval rates in the fixed DTI group order. 
    groups,totals,approved_counts,rates=calculate_approval_rates(extracted_data)

    # Convert valid application counts to a Numpy array for savinf and comparison. 
    size=np.array(totals)

    print("=======================Visualization=======================")

   # Plot approval rates in the same DTI group order used in the calcultaion.
    plt.figure(figsize=(10, 6))


    plt.plot(groups, rates, marker='o', linewidth=2.5, markersize=8, color='#4C72B0')


    # Skip empty groups whose approval rate is None.
    for i in range(len(groups)):
        if rates[i] != None:
            label_text = f"{rates[i]*100:.1f}%\n(n={totals[i]})"

            # Place each label slightly above its plotted point.
            plt.text(i, rates[i] + 0.03, label_text, ha='center', va='bottom', fontsize=10, fontweight='bold')


    plt.title("DTI Bin vs Loan Approval Rate", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("DTI Bin", fontsize=12)
    plt.ylabel("Approval Rate", fontsize=12)

    # Display proportions as percentages on the vertical axis.
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Leave space above the points for their labels.
    plt.ylim(-0.05, 1.1) 
    plt.grid(True, linestyle='--', alpha=0.6)


    plt.tight_layout()
    plt.show()

    print("=======================Analyze Complete=======================")
    print("The chart shows loan approval rates across six debt-to-income (DTI) categories.")
    print("Each data point displays:") 
    print("- Approval rate (%)")
    print("- Number of valid applications (n)")
    print("You can save these results for comparison with another state using the same loan type.")

    # Ask whether to store the results in memory for later comparison.
    save_dataYN = input("would you like to save the result? Y/N")
    if (save_dataYN == "Y" or save_dataYN == "y"):
        data_saver(state, loan_type, groups, rates, size)

def cramer_v(chi2, n):
    return (chi2/n) ** 0.5

# Compare approval outcomes for two saved states with the same loan type.
def comparison_analysis(state_1, state_2, loan_type):
    print("=======================comparison analysis=======================")
    state_1_array = []
    state_2_array = []
    significant_bin_list = []
    not_significant_bin_list = []
    cramer_V_list = []

    # Retrieve each state's saved results for the selected loan type.
    # Columns are: Dti group, approval rate, valid application count.
    state_1_array = list_object_to_array_and_filter(saved_datasets_v2.values(), state_1, loan_type)
    state_2_array = list_object_to_array_and_filter(saved_datasets_v2.values(), state_2, loan_type)

    # Calculate approved application counts from approval rates and group sizes.
    # Convert undefined products for empty groups to zero.
    state_1_passing_freq = np.nan_to_num(state_1_array[:,1].astype(float) * state_1_array[:,2].astype(float))
    state_1_failed_freq = state_1_array[:,2].astype(float) - state_1_passing_freq

    # Build a table with DTI groups as rows.
    # The two columns contain approved and non-approved application counts.
    chi_table_1 = np.column_stack((state_1_passing_freq, state_1_failed_freq))

    # Test the association between DTI group and approval outcome in state 1.
    try:
        chi2_1, p_1, dof_1, expected_1 = chi2_contingency(chi_table_1)
        print(f"======================={state_1} DTI and approval status=======================")
        print("=======================Chi-squre Test of Independence=======================")
        print(f"chi2 {state_1}", chi2_1)
        if (p_1 < 0.05):
            print(f"p value: {p_1}")
            print("In this test the p value is smaller than 0.05, we REJECTED the null hypothesis that DTI bin and approval rates are independent, meaning they have some form of statistically significant connections between them")
        else:
            print("In this test the p value is larger than 0.05, we CANNOT reject the null hypothesis that DTI bin and approval rates are independent")
    except ValueError:
        print("Insufficient data!")
        return

    # Prepare the same application counts for state 2.
    state_2_passing_freq = np.nan_to_num(state_2_array[:,1].astype(float) * state_2_array[:,2].astype(float))
    state_2_failed_freq = state_2_array[:,2].astype(float) - state_2_passing_freq

    chi_table_2 = np.column_stack((state_2_passing_freq, state_2_failed_freq))

    # Test the association between DTI group and approval outcome in state 2.
    try:
        chi2_2, p_2, dof_2, expected_2 = chi2_contingency(chi_table_2)
        print(f"======================={state_2} DTI and approval status=======================")
        print("=======================Chi-squre Test of Independence=======================")
        print(f"chi2 {state_2}", chi2_2)
        if (p_2 < 0.05):
            print(f"p value: {p_2}")
            print("In this test the p value is smaller than 0.05, we REJECTED the null hypothesis that DTI bin and approval rates are independent, meaning they have some form of statistically significant connections between them")
        else:
            print("In this test the p value is larger than 0.05, we CANNOT REJECT the null hypothesis that DTI bin and approval rates are independent")
    except ValueError:
        print("Insufficient data!")
        return

    # Combine all DTI groups to obtain each state's overall application counts.
    print(f"======================={state_1} and {state_2} Chi-square Test of Homogeneity=======================")
    state_1_passing_sum = state_1_passing_freq.sum()
    state_1_failed_sum = state_1_failed_freq.sum()
    state_2_passing_sum = state_2_passing_freq.sum()
    state_2_failed_sum = state_2_failed_freq.sum()

    # Build a 2-by-2 table.
    # Rows represent the two states; columns represent approved count and unapproved count
    homogeneity_table = np.array([[ state_1_passing_sum, state_1_failed_sum],[state_2_passing_sum, state_2_failed_sum]])

    chi_h, p_h, dof_h, expected_h = chi2_contingency(homogeneity_table)

    # Display and Interpret
    print(f"chi2 {state_1} and {state_2}", chi_h)
    if (p_h < 0.05):
        print(f"p value: {p_h}")
        print("In this test the p value is smaller than 0.05, we REJECTED the null hypothesis that the overall approval rate between the two states are equal, suggest a statistically significant difference in the overall approval rates")
    else:
        print("In this test the p value is larger than 0.05, we CANNOT REJECT the null hypothesis that the overall approval rate between the two states are equal, we don't have enough evidences to say there is a significant difference")

    # Compare the two states separately within each matching DTI group.
    same_bin_test_list = []

    print(f"======================={state_1} and {state_2} Chi-square Test of Homogeneity between the DTI bins=======================")
    print("Note: To account for the effect of multiple testings, we use 0.01 significance level in this part of the testing")

    for i in range(len(state_1_passing_freq)):
        current_bin = state_1_array[i,0]
        print(f"=======================DTI bin under examination: {current_bin} =======================")
        # Build a 2-by-2 tables for current DTI group.
        same_bin_test_table = np.array([
            [state_1_passing_freq[i],state_1_failed_freq[i]], [state_2_passing_freq[i], state_2_failed_freq[i]]
        ])

        # Test whether approval proportions differ between states in this group.
        chi2_bin, p_bin, dof_bin, expected_bin = chi2_contingency(same_bin_test_table)
        print(f"chi2 {state_1} and {state_2} in the bin: {current_bin}", chi2_bin)

        # Display and Interpret
        same_bin_result = [chi2_bin, p_bin, dof_bin, expected_bin]
        same_bin_test_list.append(same_bin_result)

        if (p_bin < 0.01):
            significant_bin_list.append(current_bin)

            # if significant association, then calculate how strong is the association
            v = cramer_v(chi2_bin, state_1_passing_freq[i]+state_1_failed_freq[i]+state_2_passing_freq[i]+state_2_failed_freq[i])
            cramer_V_list.append([current_bin, v])

        else:
            not_significant_bin_list.append(current_bin)

    # Display and Intepret
    print(f"for the following DTI bins, we can REJECT the null hypothesis that the approval rate of those specific bins are equal in the two states: {state_1} and {state_2}, the approval rate of the same DTI bin in the two states are different")
    print("In the table the other column is Cramer's V, indicating how strong the association between state and approval rate is, closer to 0 indicate a more negligble association while larger than 0.5 could point to a stronger association")
    print("DTI Bin          Cramér's V")
    print("----------------------------")
    for dti_bin,v in cramer_V_list:
        print(f"{dti_bin}     {v}")

    print("=================================================================")
    print(f"for the following DTI bins, we CANNOT REJECT the null hypothesis that the approval rate of those specific bins are equal in the two states: {state_1} and {state_2}, we don't have enough evidence to say their approval rate are different in the two states")
    print(not_significant_bin_list)

    return

# Read user commands and run the requested function until the user quits.
def cli():

    while True:
        print("=======================Main Menu=======================")
        command = input("please input your command, use /help for help")

        # Analyse a single state and optionally save its resultes.
        if (command == "/analyzeSingleState"):
            try:
                #turn to 50000 if run out of memory
                analyze_state_DTI_approval_rate(200000)
            except Exception as e:
                print("analyze failed for:" ,e)
                print("returning to main")
                continue
            continue

        # Compare two states whose results have already been saved.
        elif (command == "/comparisonAnalysis"):

            # Enter the arguments required to run the analysis, if input is an empty string, then return to main
            state_1_input = input("input the first state you would like to analyze, state need to be previously saved, type /cancel to return to main")
            state_1 = check_input_valid(state_1_input, acceptable_states)
            if state_1 == "":
                continue

            state_2_input = input("input the second state you would like to analyze, state need to be previously saved, type /cancel to return to main")
            state_2 = check_input_valid(state_2_input, acceptable_states)
            if state_2 == "":
                continue

            loan_type_input=input("Enter your loan type as a integer number from 1 to 4, type /cancel if you wish to return to main, where you could check the meaning of loan types").strip()
            loan_type = check_input_valid(loan_type_input, acceptable_loan_type)
            if loan_type == "":
                continue

            try:
                comparison_analysis(state_1,state_2,loan_type) 
            except Exception as e:
                print("analyze failed for:" ,e)
                print("returning to main")
                continue
            continue    

        # perhaps you mean /exit ?
        elif (command == "/q"):
            break

        # Clear results stores in memory
        elif (command == "/flush"):
            saved_datasets_v2.clear()
            continue

        # Display the available sommands and their purpoes.
        elif (command == "/help"):
            print("available commands:")
            print("/explainLoanType  ","to print a dictionary of loantypes and their meaning")
            print("/analyzeSingleState  ","to analyze a single state, you will have the opportunity to save the result at the end of it")
            print("/comparisonAnalysis  ", "to analyze two states from saved data")
            print("/flush  ", "to clear the saved results")
            print("/q  ", "to quit")

        # Display a list explaning the different loan_types
        elif (command == "/explainLoanType"):
            print("LoanType 1 is: ","Conventional loan")
            print("LoanType 2 is: ","Federal Housing Administration (FHA)–insured loan")
            print("LoanType 3 is: ","Department of Veterans Affairs (VA)–guaranteed loan")
            print("LoanType 4 is: ","USDA Rural Housing Service (RHS) or Farm Service Agency (FSA)–guaranteed loan")

        # Instruction unclear
        else:
            print("Instruction unclear")
            continue

# Entrance, start here
def main():

    # global variables are made so that they can be accessible within the functions
    global saved_datasets_v2
    global acceptable_states
    global acceptable_loan_type

    # Initializing
    saved_datasets_v2 = {}
    acceptable_states = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY", "DC", "GU", "PR", "VI"]
    acceptable_loan_type = ["1", "2", "3", "4"]
    cli()

# start the interactive programme.
main()

