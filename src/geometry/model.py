import gmsh
import os
import numpy as np

def create_implant_model(
    length=0.160,          
    outer_radius=0.012,    
    inner_radius=0.008,    
    fracture_gap=0.002,    
    fracture_x=0.080,      
    plate_length=0.100,
    plate_width=0.016,
    plate_thickness=0.006,
    mesh_size=0.005,
    output_path="model.msh"
):
    """
    Generates a simplified fractured bone and an attached fixation plate,
    fragmented into a single conforming mesh.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ImplantModel")
    
    # ---------------------------------------------------------
    # 1. Proximal Bone
    # ---------------------------------------------------------
    proximal_len = fracture_x - fracture_gap / 2.0
    prox_outer = gmsh.model.occ.addCylinder(0, 0, 0, proximal_len, 0, 0, outer_radius)
    prox_inner = gmsh.model.occ.addCylinder(0, 0, 0, proximal_len, 0, 0, inner_radius)
    
    # ---------------------------------------------------------
    # 2. Distal Bone
    # ---------------------------------------------------------
    distal_start = fracture_x + fracture_gap / 2.0
    distal_len = length - distal_start
    dist_outer = gmsh.model.occ.addCylinder(distal_start, 0, 0, distal_len, 0, 0, outer_radius)
    dist_inner = gmsh.model.occ.addCylinder(distal_start, 0, 0, distal_len, 0, 0, inner_radius)
    
    # ---------------------------------------------------------
    # 3. Gap Tissue
    # ---------------------------------------------------------
    gap_outer = gmsh.model.occ.addCylinder(proximal_len, 0, 0, fracture_gap, 0, 0, outer_radius)
    gap_inner = gmsh.model.occ.addCylinder(proximal_len, 0, 0, fracture_gap, 0, 0, inner_radius)
    
    # ---------------------------------------------------------
    # 4. Fixation Plate
    # ---------------------------------------------------------
    # Centered at fracture_x. Placed on top of the bone (y = outer_radius)
    px = fracture_x - plate_length / 2.0
    py = outer_radius - 0.001 # embed it 1mm into the bone for clean boolean interface
    pz = -plate_width / 2.0
    
    plate = gmsh.model.occ.addBox(px, py, pz, plate_length, plate_thickness + 0.001, plate_width)
    
    # ---------------------------------------------------------
    # Fragment all together to ensure conforming mesh interfaces
    # ---------------------------------------------------------
    vols = [
        (3, prox_outer), (3, prox_inner),
        (3, dist_outer), (3, dist_inner),
        (3, gap_outer), (3, gap_inner),
        (3, plate)
    ]
    
    # fragment
    gmsh.model.occ.fragment(vols, [])
    gmsh.model.occ.synchronize()
    
    # Note: After fragmenting, we'd normally assign physical groups.
    # To do this automatically, we can find the center of mass of each resulting volume
    # and assign it based on its bounding box.
    vols_out = gmsh.model.getEntities(3)
    
    groups = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 10: []}
    names = {
        1: "Proximal_Cortical", 2: "Proximal_Trabecular",
        3: "Distal_Cortical", 4: "Distal_Trabecular",
        5: "Gap_Cortical", 6: "Gap_Trabecular",
        10: "Plate"
    }
    
    for tag in [v[1] for v in vols_out]:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(3, tag)
        
        # The plate extends beyond the outer_radius
        if ymax > outer_radius + 0.001:
            groups[10].append(tag)
        # Proximal region (x < 0.079)
        elif xmax <= proximal_len + 1e-4:
            # If the volume's max Y is within the inner radius, it's trabecular
            if ymax < inner_radius + 1e-4:
                groups[2].append(tag)
            else:
                groups[1].append(tag)
        # Distal region (x > 0.081)
        elif xmin >= distal_start - 1e-4:
            if ymax < inner_radius + 1e-4:
                groups[4].append(tag)
            else:
                groups[3].append(tag)
        # Gap region (0.079 < x < 0.081)
        else:
            if ymax < inner_radius + 1e-4:
                groups[6].append(tag)
            else:
                groups[5].append(tag)

    for k, v in groups.items():
        if v:
            gmsh.model.addPhysicalGroup(3, v, k, name=names[k])

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)
    
    gmsh.write(output_path)
    gmsh.finalize()
    return output_path

if __name__ == "__main__":
    create_implant_model(output_path="test_model.msh")
    print("Generated test_model.msh")
