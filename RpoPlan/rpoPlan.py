import numpy as np
import pandas as pd
import pyvista as pv
import time
import argparse
import os

def create_refined_animation(csv_path, record=False, output_path=None):
    """
    Create a refined orbital visualization with:
    - Black background
    - Route revealed step by step
    - Animation speed of 0.3 seconds per frame
    - Camera focused on the deputy spacecraft
    - Zoomed out view of the Earth
    - Smaller spacecraft markers

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing orbital data
    """
    # Enable PyVista's global theme to allow empty meshes
    pv.global_theme.allow_empty_mesh = True

    # Set the theme to use black background
    pv.set_plot_theme("dark")

    # Read the data
    print(f"Reading orbital data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows of orbital data")

    # Sample data for performance
    step_size = max(1, len(df) // 400)  # Use ~400 points for smooth animation
    df_sampled = df.iloc[::step_size].copy()
    print(f"Sampled to {len(df_sampled)} points for visualization")

    # Calculate approximate points in one orbit
    # For GEO satellites (like in this data), one orbit is approximately 24 hours
    # For LEO satellites, one orbit is approximately 90 minutes
    if 'secondsSinceStart' in df_sampled.columns:
        total_time = df_sampled['secondsSinceStart'].max() - df_sampled['secondsSinceStart'].min()
        # Assuming GEO orbit (24 hours = 86400 seconds)
        orbit_time = 86400  # seconds in 24 hours
        # Calculate points in one orbit based on time ratio
        points_in_one_orbit = int(len(df_sampled) * (orbit_time / total_time))
        # Ensure reasonable limits
        points_in_one_orbit = max(20, min(points_in_one_orbit, len(df_sampled) // 2))
        print(f"Estimated points in one orbit: {points_in_one_orbit}")
    else:
        # Default if we can't calculate
        points_in_one_orbit = len(df_sampled) // 4
        print(f"Using default points in one orbit: {points_in_one_orbit}")

    # Calculate points for two orbits (as requested)
    points_in_two_orbits = points_in_one_orbit * 2
    print(f"Using {points_in_two_orbits} points to show two orbits worth of trajectory")

    # Replace NaN values with zeros
    df_sampled = df_sampled.fillna(0)

    # Earth radius
    earth_radius = 6371  # km

    # Get spacecraft positions
    chief_pos = df_sampled[['positionChiefEciX', 'positionChiefEciY', 'positionChiefEciZ']].values
    deputy_pos = df_sampled[['positionDeputyEciX', 'positionDeputyEciY', 'positionDeputyEciZ']].values
    rel_pos = df_sampled[['positionDepRelToChiefLvlhX', 'positionDepRelToChiefLvlhY', 'positionDepRelToChiefLvlhZ']].values

    # Calculate scale factor for global view
    orbit_distances = np.sqrt(np.sum(chief_pos**2, axis=1))
    avg_orbit_radius = np.mean(orbit_distances)
    print(f"Average orbit radius: {avg_orbit_radius:.2f} km")

    # Scale factor for global view - reduced to zoom out more
    scale_factor = earth_radius * 2 / avg_orbit_radius  # Reduced from 5 to 3 for more zoom out
    print(f"Using scale factor: {scale_factor:.6f}")

    # Get time and range data
    times = df_sampled['secondsSinceStart'].values

    if 'relativeRange' in df_sampled.columns:
        # Convert relative range from meters to kilometers
        range_values = df_sampled['relativeRange'].values / 1000.0  # Convert from m to km
        print(f"Converting relative range from meters to kilometers")
    else:
        # Calculate range from position data (already in km)
        range_values = np.sqrt(np.sum(rel_pos**2, axis=1))

    # Calculate maximum relative distance for scaling
    max_rel_dist = np.max(np.sqrt(np.sum(rel_pos**2, axis=1)))
    min_rel_dist = np.min(np.sqrt(np.sum(rel_pos**2, axis=1)))
    print(f"Maximum relative distance: {max_rel_dist:.2f} km")
    print(f"Minimum relative distance: {min_rel_dist:.2f} km")

    # Base size factors for spacecraft
    base_chief_size_factor = 0.0003  # Base size factor
    base_deputy_size_factor = base_chief_size_factor * 0.75  # Deputy slightly smaller than chief

    # Scale positions for global view
    chief_orbit_points = chief_pos * scale_factor
    deputy_orbit_points = deputy_pos * scale_factor

    # Animation loop - manually step through frames
    n_frames = len(df_sampled)
    print(f"Animation will run through {n_frames} frames at 0.4 seconds per frame")
    print("Press 'q' at any time to exit the animation")

    # Set up initial plotter
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])

    # Create Earth with texture if available, otherwise solid blue
    earth = pv.Sphere(radius=earth_radius, center=(0, 0, 0))
    try:
        earth_texture = pv.examples.load_globe_texture()
        plotter.add_mesh(earth, texture=earth_texture)
    except:
        plotter.add_mesh(earth, color='blue', opacity=0.8)

    # Setup global view (subplot 0, 0)
    plotter.subplot(0, 0)
    plotter.add_text("Global View - Earth and Spacecraft Orbits", position="upper_edge",
                   font_size=10, color='white')

    # Set view for global subplot - zoom out more
    plotter.view_isometric()
    plotter.camera.zoom(0.8)  # Zoom out a bit more (values less than 1 zoom out)

    # Setup LVLH view (subplot 0, 1)
    plotter.subplot(0, 1)

    # Add title with smaller font
    plotter.add_text("Deputy-Centered View - Relative Approach", position="upper_edge",
                   font_size=10, color='white')

    # Set view for LVLH subplot
    plotter.view_isometric()

    # Set up movie recording if requested
    if record:
        if output_path is None:
            # Default output path if none provided
            output_path = os.path.join(os.path.dirname(csv_path), 'orbital_animation.mp4')
        print(f"Recording animation to {output_path}")
        # Use open_gif instead of open_movie to avoid fps parameter issues
        # We'll convert the GIF to MP4 later if needed
        gif_path = output_path.replace('.mp4', '.gif')
        plotter.open_gif(gif_path)

    # Initialize plotter
    plotter.show(auto_close=False)

    # Actors to update during animation
    chief_actor = None
    deputy_actor = None
    chief_path_actor = None
    deputy_path_actor = None
    time_text_actor = None
    range_text_actor = None
    deputy_rel_actor = None
    chief_rel_actor = None
    rel_path_actor = None

    # We're no longer using coordinate axes actors
    # This space intentionally left empty to maintain code structure

    # Animation will be controlled by the timer callback

    # Define the animation callback function
    def update_animation(frame_index):

        nonlocal chief_actor, deputy_actor, chief_path_actor, deputy_path_actor
        nonlocal deputy_rel_actor, chief_rel_actor, rel_path_actor, time_text_actor, range_text_actor
        # Removed reference to axes_actors since we're no longer using them

        # Get current frame index
        i = frame_index

        # Global view updates
        plotter.subplot(0, 0)

        # Remove old actors
        if chief_actor:
            plotter.remove_actor(chief_actor, render=False)
        if deputy_actor:
            plotter.remove_actor(deputy_actor, render=False)
        if chief_path_actor:
            plotter.remove_actor(chief_path_actor, render=False)
        if deputy_path_actor:
            plotter.remove_actor(deputy_path_actor, render=False)

        # Add current trajectory (path revealed step by step, limited to 2 orbits)
        # Use the calculated points_in_two_orbits value

        # Get the current points with a sliding window of two orbits length
        start_idx = max(0, i - points_in_two_orbits + 1)
        current_chief_points = chief_orbit_points[start_idx:i+1]
        current_deputy_points = deputy_orbit_points[start_idx:i+1]

        # Create paths up to current point with the sliding window
        if len(current_chief_points) > 1:
            chief_path = pv.PolyData(current_chief_points)
            lines = []
            for j in range(len(current_chief_points) - 1):
                lines.append([2, j, j+1])
            chief_path.lines = lines
            chief_path_actor = plotter.add_mesh(chief_path, color='blue', line_width=2, render=False)

        if len(current_deputy_points) > 1:
            deputy_path = pv.PolyData(current_deputy_points)
            lines = []
            for j in range(len(current_deputy_points) - 1):
                lines.append([2, j, j+1])
            deputy_path.lines = lines
            deputy_path_actor = plotter.add_mesh(deputy_path, color='red', line_width=2, render=False)

        # Get current relative distance for dynamic sizing
        current_range = range_values[i]

        # We'll create the spacecraft after calculating normalized_range
        # Initialize actors to None if they don't exist yet
        if chief_actor is None:
            chief_actor = None
        if deputy_actor is None:
            deputy_actor = None

        # LVLH view updates
        plotter.subplot(0, 1)

        # We're no longer using axes_actors, so no need to remove them
        # This space intentionally left empty to maintain code structure

        if deputy_rel_actor:
            plotter.remove_actor(deputy_rel_actor, render=False)
        if chief_rel_actor:
            plotter.remove_actor(chief_rel_actor, render=False)
        if rel_path_actor:
            plotter.remove_actor(rel_path_actor, render=False)
        if time_text_actor:
            plotter.remove_actor(time_text_actor, render=False)
        if range_text_actor:
            plotter.remove_actor(range_text_actor, render=False)

        # Get current positions in LVLH frame
        current_deputy_rel_pos = rel_pos[i]  # Deputy position relative to chief

        # We'll create the LVLH spacecraft after calculating normalized_range
        # Initialize actors to None if they don't exist yet
        if chief_rel_actor is None:
            chief_rel_actor = None
        if deputy_rel_actor is None:
            deputy_rel_actor = None

        # We're removing the coordinate axes (XYZ lines) from the chief as requested
        # This space intentionally left empty to maintain code structure

        # Create relative path up to current point (with sliding window for consistency)
        # Use the same sliding window approach as the global view - two orbits worth
        start_idx = max(0, i - points_in_two_orbits + 1)
        current_rel_points = rel_pos[start_idx:i+1]
        if len(current_rel_points) > 1:
            rel_path = pv.PolyData(current_rel_points)
            lines = []
            for j in range(len(current_rel_points) - 1):
                lines.append([2, j, j+1])
            rel_path.lines = lines
            rel_path_actor = plotter.add_mesh(rel_path, color='cyan', line_width=2, render=False)

        # Set the camera to focus on the deputy
        # This creates a deputy-centered view while still showing the chief
        plotter.camera.focal_point = current_deputy_rel_pos

        # Enhanced dynamic zoom based on relative range
        # Calculate normalized range on a logarithmic scale for better sensitivity

        # Define the reference range (1km) for maximum zoom level
        reference_range = 1.0  # km (was 1000m, now converted to km)

        if current_range < 0.001 or min_rel_dist < 0.001:  # Prevent log(0) errors
            normalized_range = 0.0
        else:
            # Calculate the normalized range
            if current_range < reference_range:
                # For ranges below 1000km, use a fixed normalized value
                # This will maintain the zoom level we have at 1000km
                log_ref = np.log(reference_range)
                log_min = np.log(min_rel_dist)
                normalized_range = (log_ref - log_min) / (np.log(max_rel_dist) - log_min)
            else:
                # For ranges above 1000km, use normal normalization
                normalized_range = (np.log(current_range) - np.log(min_rel_dist)) / (np.log(max_rel_dist) - np.log(min_rel_dist))

            # Handle potential NaN or infinity values
            if np.isnan(normalized_range) or np.isinf(normalized_range):
                normalized_range = 0.5  # Default to middle value
            normalized_range = max(0.0, min(1.0, normalized_range))

        # Print current range and zoom info every 50 frames for debugging
        if i % 50 == 0:
            print(f"Frame {i}: Range = {current_range:.2f} km, Normalized = {normalized_range:.3f}")

        # We'll calculate the zoom factor when setting the camera position

        # Calculate camera position with enhanced dynamic zoom but limited to 1000km reference
        # Standard positioning relative to deputy with consistent zoom level
        # Calculate zoom factor based on normalized range
        zoom_factor = 0.25 * (normalized_range ** 0.5)
        zoom_factor = max(0.15, min(0.25, zoom_factor)) 

        # Use standard positioning relative to deputy
        plotter.camera.position = current_deputy_rel_pos + np.array([max_rel_dist*zoom_factor,
                                                                  max_rel_dist*zoom_factor,
                                                                  max_rel_dist*zoom_factor])

        # Set a consistent view angle based on the 1000km reference
        # This will maintain a consistent field of view
        view_angle = 5.0  # Use a fixed view angle for consistency
        plotter.camera.view_angle = view_angle

        # Use consistent color for range text
        range_color = 'white'  # Keep text color consistent

        # Now that we have normalized_range, update spacecraft sizes
        # Scale spacecraft size inversely with distance - smaller when far, larger when close
        # Invert the scale so smaller normalized_range (closer) gives larger spacecraft
        distance_scale_factor = 1.0 - 0.7 * normalized_range
        distance_scale_factor = max(0.3, min(1.5, distance_scale_factor))  # Clamp between 0.3 and 1.5

        # Update global view spacecraft sizes
        plotter.subplot(0, 0)  # Switch to global view subplot

        if chief_actor:
            plotter.remove_actor(chief_actor, render=False)
        if deputy_actor:
            plotter.remove_actor(deputy_actor, render=False)

        # Create dynamically sized spacecraft for global view
        chief_radius = earth_radius * 0.06 * distance_scale_factor
        deputy_radius = chief_radius * 0.75

        chief_sphere = pv.Sphere(radius=chief_radius, center=chief_orbit_points[i])
        deputy_sphere = pv.Sphere(radius=deputy_radius, center=deputy_orbit_points[i])

        chief_actor = plotter.add_mesh(chief_sphere, color='cyan', render=False)
        deputy_actor = plotter.add_mesh(deputy_sphere, color='magenta', render=False)

        # Update LVLH view spacecraft sizes
        plotter.subplot(0, 1)  # Switch to LVLH view subplot

        if chief_rel_actor:
            plotter.remove_actor(chief_rel_actor, render=False)
        if deputy_rel_actor:
            plotter.remove_actor(deputy_rel_actor, render=False)

        # Create dynamically sized spacecraft for LVLH view
        lvlh_chief_size = max_rel_dist * base_chief_size_factor * distance_scale_factor
        lvlh_deputy_size = lvlh_chief_size * 0.75

        chief_rel_sphere = pv.Sphere(radius=lvlh_chief_size, center=(0, 0, 0))
        deputy_rel_sphere = pv.Sphere(radius=lvlh_deputy_size, center=current_deputy_rel_pos)

        chief_rel_actor = plotter.add_mesh(chief_rel_sphere, color='blue', render=False)
        deputy_rel_actor = plotter.add_mesh(deputy_rel_sphere, color='red', render=False)

        # Update text information with better spacing to avoid overlap
        current_time = times[i]
        time_pct = ((current_time - times[0]) / (times[-1] - times[0])) * 100

        # Position text at the bottom of the screen
        # Mission time at bottom left - convert seconds to hours (with decimal)
        hours = current_time / 3600  # Convert to hours with decimal precision

        # Format time as hours with 2 decimal places
        formatted_time = f"{hours:.2f}"

        time_text_actor = plotter.add_text(f"Mission Time: {formatted_time} hours - {time_pct:.1f}% complete",
                                        position=(0.02, 0.06), font_size=10, color='white',
                                        viewport=True, render=False)

        # Range text with color based on proximity - positioned to the right of mission time
        range_text_actor = plotter.add_text(f"Relative Range: {current_range:.2f} km",
                                         position=(0.02, 0.02), font_size=10, color=range_color,
                                         viewport=True, render=False)

        # Add frame counter at the bottom right
        frame_counter = plotter.add_text(f"Frame {i+1}/{n_frames}", position=(0.85, 0.02),
                                       font_size=10, color='white', viewport=True, render=False)

        # Return the frame counter actor so we can remove it in the next callback
        return frame_counter

    # Define the timer callback function
    def timer_callback(step):

        # Use the step as our frame index
        i = step
        # Update the animation for the current frame
        frame_counter = update_animation(i)
        # Write the current frame to the GIF file if recording is enabled
        if record:
            plotter.write_frame()  # This works with open_gif

        # Schedule removal of frame counter in the next render cycle
        if frame_counter:
            plotter.remove_actor(frame_counter, render=False)

    # Set up the timer event - 400ms per frame (slowed down significantly as requested)
    # Duration is in milliseconds
    plotter.add_timer_event(max_steps=n_frames, duration=400, callback=timer_callback)

    # Start the interactive rendering loop
    try:
        plotter.show()
    except Exception as e:
        print(f"Animation stopped: {e}")
    finally:
        if record:
            print(f"Animation recording complete. Saved to {gif_path}")
            print("You can convert the GIF to MP4 using a tool like FFmpeg if needed.")
        print("Animation complete.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create orbital animation from CSV data')
    parser.add_argument('--csv', default="../CelestialChoreography/Data/RpoPlan.csv",
                        help='Path to the CSV file with orbital data')
    parser.add_argument('--record', action='store_true',
                        help='Record the animation to a video file')
    parser.add_argument('--output', default=None,
                        help='Output path for the recorded video (default: orbital_animation.mp4 in the same directory as the CSV)')
    args = parser.parse_args()

    # Create the animation
    create_refined_animation(args.csv, record=args.record, output_path=args.output)