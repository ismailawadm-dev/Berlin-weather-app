st.subheader("2) Imminent rain (next 10–30 min)")
if st.button("Check now"):
    with st.spinner("Fetching DWD radar and nowcasting…"):
        try:
            rx = fetch_radar_last_hour()  # dBZ xarray DataArray
        except Exception as e:
            st.error(f"Radar fetch failed: {e}")
            st.stop()
        R = reflectivity_to_rainrate(rx.values)
        import pysteps as ps
        oflow = ps.motion.get_method("lucaskanade")(R)
        extrap = ps.extrapolation.get_method("semilagrangian")
        steps = max(1, int(round(lead_minutes/5)))
        Rf = extrap(R[-12:], oflow, steps)
        j,i = berlin_point_index(rx, lat, lon)
        rain_now = float(R[-1, j, i])
        rain_future = float(Rf[steps-1, j, i])
        st.metric("Rain now (mm/h)", f"{rain_now:.2f}")
        st.metric(f"Rain in +{lead_minutes} min (mm/h)", f"{rain_future:.2f}")
