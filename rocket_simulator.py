import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Kulfi Model Rocket Simulator", layout="centered")

st.sidebar.title("Kulfi Rocket Simulator")
mode = st.sidebar.radio("Choose Mode", ["Advanced Mode", "Simplified Mode"])


# --------- SIMULATION FUNCTION ----------
def simulate_flight(mass, thrust, burn_time, drag_coefficient, diameter, launch_angle_deg=90,
                    wind_speed=0, parachute_altitude=30, flammability="Medium",
                    stages=None, gravity_multiplier=1.0, chaos_factor=0.0):
    """
    Simulates the flight of a model rocket, supporting single-stage (Baby Mode),
    multi-stage (Advanced Mode), and a simplified "Simplified Mode" with limited inputs.

    Args:
        mass (float): Initial mass of the rocket (kg). In Advanced Mode, this is the
                      sum of all stage masses.
        thrust (float): Engine thrust (N). Used for Baby Mode.
        burn_time (float): Engine burn time (s). Used for Baby Mode.
        drag_coefficient (float): Coefficient of drag.
        diameter (float): Rocket diameter (m).
        launch_angle_deg (int): Launch angle from horizontal (degrees).
        wind_speed (int): Horizontal wind speed (m/s).
        parachute_altitude (int): Altitude at which parachute deploys (m).
        flammability (str): Flammability risk ("Low", "Medium", "High").
        stages (list): List of dictionaries for Advanced Mode, where each dict
                       has keys 'mass', 'thrust', 'burn' for each stage.
        gravity_multiplier (float): Multiplier for gravity (Simplified Mode, defaults to 1.0).
        chaos_factor (float): Factor influencing minor random elements (Simplified Mode, defaults to 0.0).

    Returns:
        dict: A dictionary containing simulation results and messages, including:
              - 'max_altitude' (float): Maximum altitude reached (m).
              - 'flight_time' (float): Total flight duration (s).
              - 'horizontal_distance' (float): Total horizontal distance traveled (s).
              - 'feasible' (bool): True if simulation was feasible, False otherwise.
              - 'launch_msgs' (list): List of warning or error messages.
              - 'fire_risk' (str): Calculated fire risk.
              - 'time' (list): List of time points during flight.
              - 'altitude' (list): List of altitude points during flight.
              - 'velocity' (list): List of velocity points during flight.
              - 'horizontal_pos' (list): List of horizontal position points during flight.
              - 'apogee_index' (int): Index in time/altitude lists corresponding to max altitude.
              - 'stage_separation_times' (list): Times at which stages separated (Advanced Mode).
              - 'parachute_deploy_time' (float): Time at which parachute deployed.
              - 'parachute_deploy_altitude' (float): Altitude at which parachute deployed.
    """

    # Physics constants
    g = 9.81 * gravity_multiplier
    rho = 1.225
    A = np.pi * (diameter / 2) ** 2
    dt = 0.01 if mode == "Advanced Mode" else (0.05 if mode == "Simplified Mode" else 0.05)

    launch_angle_rad = np.radians(launch_angle_deg)

    # Initialize lists to store flight data
    time_list = [0]
    velocity = [0]
    altitude = [0]
    horizontal_pos = [0]

    t, v, h, x = 0, 0, 0, 0  # Current time, velocity, altitude, horizontal position
    stage_separation_times = []

    # Variables to record parachute deployment
    parachute_deploy_time = None
    parachute_deploy_altitude = None

    # List to collect warnings and error messages
    launch_msgs = []

    # --------- GENERAL INPUT VALIDATION (applies to all modes) ---------
    if diameter <= 0:
        return {"feasible": False, "launch_msgs": ["Rocket diameter must be positive."]}
    if drag_coefficient < 0:
        return {"feasible": False, "launch_msgs": ["Drag coefficient cannot be negative."]}
    if parachute_altitude < 0:
        launch_msgs.append("Parachute altitude cannot be negative. Setting to 0.")
        parachute_altitude = 0

    # --------- MODE-SPECIFIC VALIDATIONS AND SIMULATION LOGIC ---------
    if mode == "Simplified Mode":
        if mass <= 0 or thrust <= 0 or burn_time <= 0:
            return {"feasible": False, "launch_msgs": ["Mass, thrust, and burn time must be positive."]}

        # Thrust-to-weight ratio check
        if thrust / (mass * g) < 1.1:
            return {"feasible": False, "launch_msgs": ["Thrust-to-weight ratio too low. Rocket won't lift off."]}

        # Warnings
        if launch_angle_deg < 45 or launch_angle_deg > 90:
            launch_msgs.append("Unusual launch angle -- generally 45-90 is recommended.")
        if wind_speed > 10:
            launch_msgs.append("High wind speed -- stability may be compromised.")
        
        launch_msgs.append("Rocket is launching.")
        wind_speed = wind_speed + (np.random.rand() - 0.5) * chaos_factor * 2
        parachute_altitude = parachute_altitude + (np.random.rand() - 0.5) * chaos_factor * 10


        # --- Single-stage simulation ---
        # Powered flight phase
        while t < burn_time:
            F_thrust = thrust
            F_drag = 0.5 * rho * drag_coefficient * A * v ** 2 * np.sign(v)
            F_net = F_thrust - F_drag - mass * g
            a = F_net / mass

            v += a * dt
            h += v * dt * np.sin(launch_angle_rad)
            x += v * dt * np.cos(launch_angle_rad)
            t += dt

            time_list.append(t)
            velocity.append(v)
            altitude.append(h)
            horizontal_pos.append(x)

            if h < 0:
                h = 0
                break

        # Coasting phase (after burn, before apogee)
        while v > 0 and h >= 0:
            F_drag = 0.5 * rho * drag_coefficient * A * v ** 2 * np.sign(v)
            F_net = -F_drag - mass * g
            a = F_net / mass

            v += a * dt
            h += v * dt * np.sin(launch_angle_rad)
            x += v * dt * np.cos(launch_angle_rad)
            t += dt

            time_list.append(t)
            velocity.append(v)
            altitude.append(h)
            horizontal_pos.append(x)

            if h < 0:
                h = 0
                break

    elif mode == "Advanced Mode": # Advanced Mode: Multi-Stage Rocket Simulation
        if not stages or not isinstance(stages, list) or len(stages) == 0:
            return {"feasible": False, "launch_msgs": ["Advanced Mode requires at least one stage configuration."]}

        total_initial_mass = 0
        for i, stage in enumerate(stages):
            # Validate each stage's parameters
            if not all(k in stage for k in ["mass", "thrust", "burn"]):
                return {"feasible": False, "launch_msgs": [f"Stage {i+1} is missing 'mass', 'thrust', or 'burn' parameters."]}
            # Thrust and burn can be 0 for a non-propulsive payload/recovery stage
            if stage["mass"] < 0: # Mass cannot be negative
                    return {"feasible": False, "launch_msgs": [f"Stage {i+1} mass cannot be negative."]}
            if stage["thrust"] < 0 or stage["burn"] < 0:
                return {"feasible": False, "launch_msgs": [f"Stage {i+1} thrust and burn time cannot be negative."]}
            total_initial_mass += stage["mass"]

        # Advanced Mode specific warnings/cancellations
        if launch_angle_deg < 60 or launch_angle_deg > 90:
            launch_msgs.append("For multi-stage rockets, a launch angle between 60 and 90 is generally recommended for optimal performance and recovery.")
        if wind_speed > 7:
            launch_msgs.append("High wind speed (above 7 m/s) detected. This can significantly impact multi-stage rocket trajectory and recovery.")
        if parachute_altitude > 1000:
            launch_msgs.append("Parachute deployment altitude seems very high. Ensure it's realistic for your rocket's apogee.")

        if stages[0]["mass"] <= 0 or stages[0]["thrust"] / (stages[0]["mass"] * g) < 1.5:
            return {"feasible": False, "launch_msgs": ["First stage thrust-to-weight ratio is too low (recommended > 1.5 for advanced rockets) or mass is zero. Rocket may not clear the launch tower or achieve stable flight."]}

        current_rocket_mass = total_initial_mass
        
        # Iterate through each stage for powered flight
        for i, stage in enumerate(stages):
            burn_time_stage = stage["burn"]
            thrust_stage = stage["thrust"]
            
            # Calculate the mass of the rocket *after* this stage burns out and separates
            # This is the mass of all subsequent stages + payload (if any)
            mass_after_this_stage = sum(s["mass"] for s in stages[i+1:])

            stage_burn_start_time = t
            stage_burn_end_time = stage_burn_start_time + burn_time_stage

            # Simulate burn for this stage only if it has thrust and burn time
            if burn_time_stage > 0 and thrust_stage > 0:
                while t < stage_burn_end_time:
                    F_drag = 0.5 * rho * drag_coefficient * A * v ** 2 * np.sign(v)
                    F_net = thrust_stage - F_drag - current_rocket_mass * g
                    a = F_net / current_rocket_mass

                    v += a * dt
                    h += v * dt * np.sin(launch_angle_rad)
                    x += v * dt * np.cos(launch_angle_rad)
                    t += dt

                    time_list.append(t)
                    velocity.append(v)
                    altitude.append(h)
                    horizontal_pos.append(x)
                    
                    if h < 0:
                        h = 0
                        break
            
            # If this was a propulsive stage that completed its burn, record separation and update mass
            # or if it's a non-propulsive stage that completes its "burn time" (i.e. we move to next stage's mass)
            if t >= stage_burn_end_time: # Changed condition to handle 0 burn time stages correctly for mass transition
                if burn_time_stage > 0 or thrust_stage > 0: # Only mark separation if it was an active stage
                    stage_separation_times.append(t)
                current_rocket_mass = mass_after_this_stage
            
            if h <= 0:
                h = 0
                break

        # Correctly determine final_rocket_mass for descent
        final_rocket_mass = stages[len(stages) - 1]["mass"] if stages else 0.1 
        
        # Check if the final mass for descent is zero or negative
        if final_rocket_mass <= 0:
            launch_msgs.append(
                "Rocket mass for descent is zero or negative, parachute phase may not simulate correctly. "
                "Defaulting to 0.1kg for descent."
            )
            final_rocket_mass = 0.1 # Assign a small positive mass to prevent division by zero or unrealistic behavior

        # Continue coasting as long as there's upward vertical velocity and rocket is above ground
        while v > 0 and h >= 0:
            F_drag = 0.5 * rho * drag_coefficient * A * v ** 2 * np.sign(v)
            F_net = -F_drag - final_rocket_mass * g
            a = F_net / final_rocket_mass

            v += a * dt
            h += v * dt * np.sin(launch_angle_rad)
            x += v * dt * np.cos(launch_angle_rad)
            t += dt

            time_list.append(t)
            velocity.append(v)
            altitude.append(h)
            horizontal_pos.append(x)
            
            if h < 0:
                h = 0
                break
        
    # Calculate max altitude and its index
    max_altitude = max(altitude) if altitude else 0
    apogee_index = altitude.index(max_altitude) if altitude else 0

    # Parachute descent phase
    parachute_deployment_mass = final_rocket_mass if mode == "Advanced Mode" else mass 

    parachute_deployed = False
    max_simulation_steps = 1000000
    step_count = 0

    while h > 0 and step_count < max_simulation_steps:
        step_count += 1
        
        # Deploy parachute if altitude is at or below deployment altitude AND rocket is descending (or at apogee)
        if not parachute_deployed and (h <= parachute_altitude or v < 0.1): 
            parachute_drag_coeff = 1.5
            parachute_area = 1.5      

            parachute_deployed = True
            parachute_deploy_time = t # Capture deployment time
            parachute_deploy_altitude = h # Capture deployment altitude

            if mode == "Simplified Mode" and np.random.rand() < chaos_factor:
                v += (np.random.rand() - 0.5) * 5 # Add a little random 'thump' for chaos

        if parachute_deployed:
            # When parachute is deployed, use its specific drag area and the remaining mass
            F_drag = 0.5 * rho * parachute_drag_coeff * parachute_area * abs(v) * v
        else:
            # Before parachute, use rocket's initial drag area (or what's left of it)
            F_drag = 0.5 * rho * drag_coefficient * A * abs(v) * v

        F_net = -F_drag - parachute_deployment_mass * g # Use parachute_deployment_mass for descent
        a = F_net / parachute_deployment_mass

        v += a * dt
        h += v * dt
        
        if mode == "Simplified Mode":
            x += (wind_speed + (np.random.rand() - 0.5) * chaos_factor * 2) * dt
        else:
            x += wind_speed * dt
        t += dt

        if h < 0:
            h = 0
            break

        time_list.append(t)
        velocity.append(v)
        altitude.append(h)
        horizontal_pos.append(x)
        
    if step_count >= max_simulation_steps:
        launch_msgs.append("Simulation exceeded maximum time steps. This might indicate a very slow descent or an issue with parameters.")
        if mode == "Simplified Mode":
            launch_msgs.append("Rocket is taking a long time to land.")


    flight_time = time_list[-1] if time_list else 0
    horizontal_distance = horizontal_pos[-1] if horizontal_pos else 0

    # --------- FIRE RISK CALCULATION ---------
    fire_risk = "Low"
    risk_score = 0
    flammability_map = {"Low": 0, "Medium": 1, "High": 2}
    risk_score += flammability_map.get(flammability, 1)

    if wind_speed > 5:
        risk_score += 1

    if mode == "Simplified Mode":
        if thrust > 15:
            risk_score += 1
    elif mode == "Advanced Mode":
        total_thrust_adv = sum(s["thrust"] for s in stages if s["burn"] > 0) if stages else 0 # Sum only propulsive thrust
        if total_thrust_adv > 50:
            risk_score += 1

    if risk_score >= 4:
        fire_risk = "High"
    elif risk_score >= 2:
        fire_risk = "Medium"
    
    if mode == "Simplified Mode":
        if fire_risk == "High":
            fire_risk = "High Risk"
        elif fire_risk == "Medium":
            fire_risk = "Medium Risk"
        else:
            fire_risk = "Low Risk"


    return {
        "max_altitude": max_altitude,
        "flight_time": flight_time,
        "horizontal_distance": horizontal_distance,
        "feasible": True,
        "launch_msgs": launch_msgs,
        "fire_risk": fire_risk,
        "time": time_list,
        "altitude": altitude,
        "velocity": velocity,
        "horizontal_pos": horizontal_pos,
        "apogee_index": apogee_index,
        "stage_separation_times": stage_separation_times,
        "parachute_deploy_time": parachute_deploy_time,      # New return value
        "parachute_deploy_altitude": parachute_deploy_altitude # New return value
    }


# --------- STREAMLIT UI LAYOUT ---------
st.header("Kulfi Model Rocket Simulator")
st.write("Simulate your rocket's flight path and analyze performance.")


if mode == "Advanced Mode": # Advanced Mode
    st.subheader("Advanced Mode: Multi-Stage Rockets (ONLY)")

    num_stages = st.number_input("Number of Stages", min_value=1, max_value=5, value=2, key="num_stages")
    stages_config = []
    for i in range(num_stages):
        stage_title = f"Stage {i+1} Configuration"
        is_final_stage = (i == num_stages - 1)
        if is_final_stage:
            stage_title += " (Stage where Parachute Deploys)"
        
        st.markdown(f"#### {stage_title}")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            stage_mass = st.number_input(
                f"Mass (kg) - Stage {i+1}", 
                min_value=0.0 if is_final_stage else 0.01, # min_value 0.0 for final stage
                value=0.0 if is_final_stage else 0.2, 
                key=f"mass_{i}"
            )
        with col_s2:
            stage_thrust = st.number_input(
                f"Thrust (N) - Stage {i+1}", 
                min_value=0.0, 
                value=0.0 if is_final_stage else 20.0, 
                key=f"thrust_{i}"
            )
        with col_s3:
            stage_burn = st.number_input(
                f"Burn Time (s) - Stage {i+1}", 
                min_value=0.0, 
                value=0.0 if is_final_stage else 1.5, 
                key=f"burn_{i}"
            )
        stages_config.append({"mass": stage_mass, "thrust": stage_thrust, "burn": stage_burn})

    st.subheader("Rocket & Environmental Factors")
    col_adv1, col_adv2 = st.columns(2)
    with col_adv1:
        diameter_adv = st.slider("Rocket Diameter (m)", 0.05, 0.5, 0.15, 0.01, key="diameter_adv")
        drag_coefficient_adv = st.slider("Drag Coefficient", 0.1, 1.0, 0.4, 0.05, key="drag_coeff_adv")
    with col_adv2:
        launch_angle_adv = st.slider("Launch Angle (degrees from horizontal)", 60, 90, 90, 1, key="launch_angle_adv")
        wind_speed_adv = st.slider("Wind Speed (m/s)", 0, 20, 2, 1, key="wind_speed_adv")
    
    st.subheader("Recovery & Risk")
    st.info("""
    **Experiencing 'Rocket mass for descent is zero' warning?**
    In multi-stage rockets, ensure your **final stage** (the one that deploys the parachute) has a **positive mass** but its **Thrust and Burn Time are set to 0**. This signifies the part of the rocket that remains and descends.
    """)

    col_adv3, col_adv4 = st.columns(2)
    with col_adv3:
        # Changed to number_input
        parachute_altitude_adv = st.number_input(
            "Parachute Deployment Altitude (m)", 
            min_value=10.0, 
            max_value=500.0, 
            value=50.0, 
            step=5.0, 
            key="parachute_alt_adv"
        )
    with col_adv4:
        flammability_adv = st.selectbox("Flammability Risk", ["Low", "Medium", "High"], key="flammability_adv")


    if st.button("Simulate Advanced Flight"):
        total_mass_for_initial_check = sum(s["mass"] for s in stages_config)
        
        results_adv = simulate_flight(total_mass_for_initial_check, 1, 1, drag_coefficient_adv, diameter_adv,
                                      launch_angle_deg=launch_angle_adv, wind_speed=wind_speed_adv,
                                      parachute_altitude=parachute_altitude_adv, flammability=flammability_adv,
                                      stages=stages_config)
        
        if not results_adv["feasible"]:
            st.error(" ".join(results_adv["launch_msgs"]))
        else:
            if results_adv["launch_msgs"]:
                for msg in results_adv["launch_msgs"]:
                    st.warning(msg)

            st.subheader("Advanced Flight Results")
            st.metric("Max Altitude", f"{results_adv['max_altitude']:.2f} m")
            st.metric("Total Flight Time", f"{results_adv['flight_time']:.2f} s")
            st.metric("Horizontal Distance Traveled", f"{results_adv['horizontal_distance']:.2f} m")
            st.metric("Fire Risk", results_adv["fire_risk"])

            fig_adv, ax_adv = plt.subplots(figsize=(10, 6))
            ax_adv.plot(results_adv["time"], results_adv["altitude"], label="Altitude (m)")
            
            if results_adv["altitude"] and results_adv["time"]:
                apogee_time_adv = results_adv["time"][results_adv["apogee_index"]]
                apogee_altitude_adv = results_adv["altitude"][results_adv["apogee_index"]]
                ax_adv.plot(apogee_time_adv, apogee_altitude_adv, 'ro', markersize=8, label=f'Apogee: {apogee_altitude_adv:.2f}m at {apogee_time_adv:.2f}s')
                ax_adv.annotate(f'Apogee\n({apogee_altitude_adv:.2f}m)',
                                (apogee_time_adv, apogee_altitude_adv),
                                textcoords="offset points", xytext=(0,10), ha='center',
                                arrowprops=dict(facecolor='black', shrink=0.05))

            # Plot parachute deployment point
            if results_adv["parachute_deploy_time"] is not None:
                ax_adv.plot(results_adv["parachute_deploy_time"], results_adv["parachute_deploy_altitude"],
                            'g^', markersize=8, label=f'Parachute Deployed: {results_adv["parachute_deploy_altitude"]:.2f}m at {results_adv["parachute_deploy_time"]:.2f}s')
                ax_adv.annotate(f'Parachute\n({results_adv["parachute_deploy_altitude"]:.2f}m)',
                                (results_adv["parachute_deploy_time"], results_adv["parachute_deploy_altitude"]),
                                textcoords="offset points", xytext=(0,-20), ha='center',
                                arrowprops=dict(facecolor='green', shrink=0.05))

            for sep_time in results_adv["stage_separation_times"]:
                ax_adv.axvline(x=sep_time, color='r', linestyle='--', label=f'Stage Sep. at {sep_time:.2f}s' if sep_time == results_adv["stage_separation_times"][0] else '')

            ax_adv.set_xlabel("Time (s)")
            ax_adv.set_ylabel("Altitude (m)")
            ax_adv.set_title("Altitude vs. Time (Ascent & Descent - Advanced)")
            ax_adv.grid(True)
            ax_adv.legend()
            st.pyplot(fig_adv)

            fig_path_adv, ax_path_adv = plt.subplots(figsize=(10, 6))
            ax_path_adv.plot(results_adv["horizontal_pos"], results_adv["altitude"], label="Flight Path")
            ax_path_adv.set_xlabel("Horizontal Distance (m)")
            ax_path_adv.set_ylabel("Altitude (m)")
            ax_path_adv.set_title("Flight Path (Advanced)")
            ax_path_adv.grid(True)
            ax_path_adv.legend()
            st.pyplot(fig_path_adv)

            fig_vel_adv, ax_vel_adv = plt.subplots(figsize=(10, 6))
            ax_vel_adv.plot(results_adv["time"], results_adv["velocity"], label="Velocity (m/s)")
            ax_vel_adv.set_xlabel("Time (s)")
            ax_vel_adv.set_ylabel("Velocity (m/s)")
            ax_vel_adv.set_title("Velocity vs. Time (Ascent & Descent - Advanced)")
            ax_vel_adv.grid(True)
            ax_vel_adv.legend()
            st.pyplot(fig_vel_adv)

else: # Simplified Mode
    st.subheader("Simplified Mode: My First Rocket")
    st.write("Push the buttons and see what happens. Your inputs here are simplified for fun.")

    col_stupid1, col_stupid2 = st.columns(2)
    with col_stupid1:
        mass_stupid = st.slider("Rocket Weight (heavy/light)", 0.1, 2.0, 0.5, 0.1, key="mass_stupid")
        thrust_stupid = st.slider("Rocket Power (big push/little push)", 1.0, 30.0, 10.0, 1.0, key="thrust_stupid")
    with col_stupid2:
        burn_time_stupid = st.slider("Burny Time (short/long)", 0.5, 5.0, 2.0, 0.1, key="burn_time_stupid")
        diameter_stupid = st.slider("Rocket Roundness (small/big)", 0.05, 0.2, 0.1, 0.01, key="diameter_stupid")
    
    default_drag_coefficient = 0.5
    default_launch_angle = 90
    default_wind_speed = 0
    default_parachute_altitude = 30 # Simplified mode default is still fixed, but simulation still tracks deployment point
    default_flammability = "Low"
    default_gravity_multiplier = 1.0
    default_chaos_factor = 0.1


    if st.button("Launch My Rocket"):
        with st.spinner("Calculating flight path..."):
            results_stupid = simulate_flight(mass=mass_stupid, 
                                             thrust=thrust_stupid, 
                                             burn_time=burn_time_stupid, 
                                             drag_coefficient=default_drag_coefficient, 
                                             diameter=diameter_stupid,
                                             launch_angle_deg=default_launch_angle, 
                                             wind_speed=default_wind_speed,
                                             parachute_altitude=default_parachute_altitude, 
                                             flammability=default_flammability,
                                             gravity_multiplier=default_gravity_multiplier, 
                                             chaos_factor=default_chaos_factor)
            
            if not results_stupid["feasible"]:
                st.error(" ".join(results_stupid["launch_msgs"]))
            else:
                if results_stupid["launch_msgs"]:
                    for msg in results_stupid["launch_msgs"]:
                        st.warning(msg)

                st.subheader("Rocket Went...")
                st.metric("Highest Point", f"{results_stupid['max_altitude']:.2f} m")
                st.metric("Time in Air", f"{results_stupid['flight_time']:.2f} s")
                st.metric("Landing Distance", f"{results_stupid['horizontal_distance']:.2f} m")
                st.metric("Risk of Bad Thing Happening", results_stupid["fire_risk"])

                fig_stupid, ax_stupid = plt.subplots(figsize=(10, 6))
                ax_stupid.plot(results_stupid["time"], results_stupid["altitude"], label="Up and Down")
                if results_stupid["altitude"] and results_stupid["time"]:
                    apogee_time_stupid = results_stupid["time"][results_stupid["apogee_index"]]
                    apogee_altitude_stupid = results_stupid["altitude"][results_stupid["apogee_index"]]
                    ax_stupid.plot(apogee_time_stupid, apogee_altitude_stupid, 'ro', markersize=8, label=f'Highest Point: {apogee_altitude_stupid:.2f}m')
                    ax_stupid.annotate(f'Highest Point\n({apogee_altitude_stupid:.2f}m)',
                                        (apogee_time_stupid, apogee_altitude_stupid),
                                        textcoords="offset points", xytext=(0,10), ha='center',
                                        arrowprops=dict(facecolor='black', shrink=0.05))

                # Plot parachute deployment point for simplified mode
                if results_stupid["parachute_deploy_time"] is not None:
                    ax_stupid.plot(results_stupid["parachute_deploy_time"], results_stupid["parachute_deploy_altitude"],
                                 'g^', markersize=8, label=f'Parachute Deployed: {results_stupid["parachute_deploy_altitude"]:.2f}m')
                    ax_stupid.annotate(f'Parachute\n({results_stupid["parachute_deploy_altitude"]:.2f}m)',
                                        (results_stupid["parachute_deploy_time"], results_stupid["parachute_deploy_altitude"]),
                                        textcoords="offset points", xytext=(0,-20), ha='center',
                                        arrowprops=dict(facecolor='green', shrink=0.05))

                ax_stupid.set_xlabel("Time")
                ax_stupid.set_ylabel("Height")
                ax_stupid.set_title("Rocket's Journey (Up & Down)")
                ax_stupid.grid(True)
                ax_stupid.legend()
                st.pyplot(fig_stupid)

                fig_path_stupid, ax_path_stupid = plt.subplots(figsize=(10, 6))
                ax_path_stupid.plot(results_stupid["horizontal_pos"], results_stupid["altitude"], label="Rocket's Path")
                ax_path_stupid.set_xlabel("Sideways")
                ax_path_stupid.set_ylabel("Up and Down")
                ax_path_stupid.set_title("Where Rocket Went")
                ax_path_stupid.grid(True)
                ax_path_stupid.legend()
                st.pyplot(fig_path_stupid)

                fig_vel_stupid, ax_vel_stupid = plt.subplots(figsize=(10, 6))
                ax_vel_stupid.plot(results_stupid["time"], results_stupid["velocity"], label="How Fast Rocket Went")
                ax_vel_stupid.set_xlabel("Time")
                ax_vel_stupid.set_ylabel("Speed")
                ax_vel_stupid.set_title("Rocket's Speed (Getting Fast & Slow)")
                ax_vel_stupid.grid(True)
                ax_vel_stupid.legend()
                st.pyplot(fig_vel_stupid)