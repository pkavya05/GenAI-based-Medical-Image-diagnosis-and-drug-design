import React, { useEffect } from 'react';
import * as $3Dmol from '3dmol';

const ProteinViewer = ({ pdbData }) => {
  useEffect(() => {
    if (pdbData) {
      const viewer = $3Dmol.createViewer('proteinViewer', {
        width: '100%',
        height: '400px', // Adjust height as needed
        backgroundColor: 'white',
      });

      // Add PDB structure to the viewer
      viewer.addModel(pdbData, 'pdb');
      viewer.setStyle({}, { stick: {} });
      viewer.zoomTo();
      viewer.render();
    }
  }, [pdbData]);

  return (
    <div
      id="proteinViewer"
      style={{
        width: '100%',  // Make sure the width takes up the full space of the parent container
        height: '400px',  // You can adjust the height here if needed
        marginTop: '20px',
        backgroundColor: '#333',  // Set a background color for the viewer's box
        borderRadius: '8px',  // Optional: To match the design theme
        display: 'flex',
        justifyContent: 'center', // Ensure content is centered
        alignItems: 'center', // Center vertically
      }}
    ></div>
  );
};

export default ProteinViewer;
