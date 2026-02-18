import { useEffect, useRef, useState } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";

import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkXMLPolyDataReader from "@kitware/vtk.js/IO/XML/XMLPolyDataReader";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkScalarBarActor from "@kitware/vtk.js/Rendering/Core/ScalarBarActor";

export default function VtkViewer({ url, fieldName = "temperature", style }) {
  const containerRef = useRef(null);
  const grwRef = useRef(null);
  const prettyLabel = {
    temperature: "Temperature (K)",
    u_mag: "Displacement |u| (m)",
    von_mises: "Von Mises (Pa)",
    velocity_mag: "Velocity |u| (m/s)",
    pressure: "Pressure (Pa)",
    }[fieldName] ?? fieldName;

  useEffect(() => {
    if (!url || !containerRef.current) return;
    const scalarName = fieldName;
    if (grwRef.current) {
      grwRef.current.delete();
      grwRef.current = null;
      containerRef.current.innerHTML = "";
    }

    const grw = vtkGenericRenderWindow.newInstance({ background: [0.04, 0.06, 0.10] });
    grw.setContainer(containerRef.current);
    grw.resize();
    grwRef.current = grw;

    const renderer = grw.getRenderer();
    const renderWindow = grw.getRenderWindow();

    const reader = vtkXMLPolyDataReader.newInstance();
    const mapper = vtkMapper.newInstance();
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    renderer.addActor(actor);

    // Scalar bar
    const scalarBar = vtkScalarBarActor.newInstance();
    // scalarBar.setTitle(fieldName);
    // Title/label API differs across vtk.js versions
    if (typeof scalarBar.setTitle === "function") {
    scalarBar.setTitle(fieldName);
    } else if (typeof scalarBar.setAxisLabel === "function") {
    // Many versions use axis label instead of title
    scalarBar.setAxisLabel(fieldName);
    } else if (typeof scalarBar.setTitleText === "function") {
    scalarBar.setTitleText(fieldName);
    } else {
    console.warn("ScalarBarActor: no title/label setter found in this vtk.js version");
    }
    renderer.addActor2D(scalarBar);
    // use prettyLabel for scalar bar
    if (typeof scalarBar.setTitle === "function") scalarBar.setTitle(prettyLabel);
    else if (typeof scalarBar.setAxisLabel === "function") scalarBar.setAxisLabel(prettyLabel);


    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
        return r.arrayBuffer();
      })
      .then((ab) => {
        reader.parseAsArrayBuffer(ab);
        const poly = reader.getOutputData(0);

        mapper.setInputData(poly);
        // ---- pick the scalar array from POINT DATA ----
        const pd = poly.getPointData();
        pd.setActiveScalars(scalarName);

        mapper.setScalarVisibility(true);
        mapper.setScalarModeToUsePointData();        // ✅ not PointFieldData
        mapper.setColorByArrayName(scalarName);
        mapper.setInterpolateScalarsBeforeMapping(true);

        // ---- set scalar range from data ----
        const scalars = pd.getScalars();             // active scalars
        if (scalars) {
        const [rmin, rmax] = scalars.getRange();
        mapper.setScalarRange(rmin, rmax);

        const lut = mapper.getLookupTable();
        lut.setRange(rmin, rmax);
        lut.build();
        } else {
        console.warn(`No point scalars found for '${scalarName}'. Available:`,
            pd.getArrays().map(a => a.getName())
        );
        }

        // Hook scalar bar to lookup table
        scalarBar.setScalarsToColors(mapper.getLookupTable());


        // Enable coloring by scalar array
        mapper.setScalarVisibility(true);
        // mapper.setScalarModeToUsePointFieldData();
        mapper.setScalarModeToUsePointData(); 
        mapper.setColorByArrayName(scalarName);
        mapper.setInterpolateScalarsBeforeMapping(true);

        // Hook scalar bar to lookup table
        scalarBar.setScalarsToColors(mapper.getLookupTable());

        renderer.resetCamera();
        renderWindow.render();
      })
      .catch(console.error);

    const onResize = () => grw.resize();
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      if (grwRef.current) {
        grwRef.current.delete();
        grwRef.current = null;
      }
    };
  }, [url, fieldName, prettyLabel]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "420px",
        borderRadius: 16,
        overflow: "hidden",
        border: "1px solid rgba(255,255,255,0.12)",
        ...style,
      }}
    />
  );
}
