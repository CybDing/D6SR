"""
Robot Loader: Converts URDF to MuJoCo-compatible MJCF with freejoint base.

This module handles:
1. Loading and parsing the URDF file
2. Converting ROS package:// paths to absolute paths
3. Adding a freejoint to base_link (allows free movement)
4. Injecting physics parameters (damping, friction, armature)
5. Saving the modified MJCF for MuJoCo simulation
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


class RobotLoader:
    """Loads and converts URDF to MuJoCo MJCF format with physics parameters."""

    def __init__(
        self,
        robot_path: Optional[str] = None,
        urdf_filename: Optional[str] = None,
        output_name: Optional[str] = None,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        robot_v1_path: Optional[str] = None,
    ):
        """
        Initialize the robot loader.

        Args:
            robot_path: Path to robot folder (e.g. robot_v0 / robot_v1).
            urdf_filename: URDF filename inside robot_path/urdf. If None, auto-detects.
            output_name: Output MJCF filename. If None, uses "<model_name>_mujoco.xml".
            config_path: YAML config path. If None, uses default robot_params.yaml.
            model_name: MJCF model name. If None, uses robot folder name.
            robot_v1_path: Backward-compatible alias for robot_path.
        """
        if robot_path is None and robot_v1_path is not None:
            robot_path = robot_v1_path
        if robot_path is None:
            # Auto-detect path relative to this file
            self.base_path = Path(__file__).parent.parent / "robot_v1"
        else:
            self.base_path = Path(robot_path)

        urdf_dir = self.base_path / "urdf"
        if urdf_filename is not None:
            self.urdf_path = urdf_dir / urdf_filename
        else:
            default_urdf = urdf_dir / "robot_v1.urdf"
            if default_urdf.exists():
                self.urdf_path = default_urdf
            else:
                candidates = sorted(urdf_dir.glob("*.urdf"))
                if not candidates:
                    raise FileNotFoundError(f"No URDF found in: {urdf_dir}")
                self.urdf_path = candidates[0]

        self.meshes_path = self.base_path / "meshes"
        self.output_path = Path(__file__).parent / "models"
        if config_path is None:
            self.config_path = Path(__file__).parent / "config" / "robot_params.yaml"
        else:
            self.config_path = Path(config_path)

        if model_name is None:
            self.model_name = self.base_path.name
        else:
            self.model_name = model_name

        if output_name is None:
            self.output_name = f"{self.model_name}_mujoco.xml"
        else:
            self.output_name = output_name

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def load_config(self) -> Dict[str, Any]:
        """Load physics configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _fix_mesh_paths(self, urdf_content: str) -> str:
        """
        Convert ROS package:// paths to absolute file paths.
        
        The original URDF uses paths like:
            package://URDF（1）.SLDASM/meshes/base_link.STL
        
        We convert these to absolute paths for MuJoCo.
        """
        # Pattern to match package:// URIs
        pattern = r'filename="package://[^/]+/meshes/([^"]+)"'
        
        def replace_path(match):
            mesh_name = match.group(1)
            abs_path = str(self.meshes_path / mesh_name)
            return f'filename="{abs_path}"'
        
        return re.sub(pattern, replace_path, urdf_content)
    
    def _add_freejoint_to_mjcf(self, mjcf_root: ET.Element) -> ET.Element:
        """
        Add a freejoint to base_link to allow free movement in space.
        
        Without this, MuJoCo fixes base_link to the world, preventing
        the robot from moving as a whole.
        """
        # Find worldbody
        worldbody = mjcf_root.find('.//worldbody')
        if worldbody is None:
            raise ValueError("No worldbody found in MJCF")
        
        # Find the base_link body (first body in worldbody hierarchy)
        base_body = worldbody.find('body')
        if base_body is None:
            raise ValueError("No body found in worldbody")
        
        # Check if freejoint already exists
        existing_joint = base_body.find("joint[@type='free']")
        if existing_joint is not None:
            return mjcf_root
        
        # Create freejoint element
        freejoint = ET.Element('freejoint')
        freejoint.set('name', 'base_freejoint')
        
        # Insert at the beginning of body
        base_body.insert(0, freejoint)
        
        return mjcf_root
    
    def _inject_physics_params(self, mjcf_root: ET.Element, config: Dict[str, Any]) -> ET.Element:
        """
        Inject physics parameters (damping, friction, armature) into joints.
        """
        joint_config = config.get('joints', {})
        default_params = joint_config.get('default', {})
        
        # Find all joints
        for joint in mjcf_root.findall('.//joint'):
            joint_name = joint.get('name', '')
            joint_type = joint.get('type', '')
            
            # Skip freejoint
            if joint_type == 'free' or joint_name == 'base_freejoint':
                continue
            
            # Get per-joint or default parameters
            params = joint_config.get(joint_name, default_params)
            
            # Apply parameters
            if 'damping' in params:
                joint.set('damping', str(params['damping']))
            if 'frictionloss' in params:
                joint.set('frictionloss', str(params['frictionloss']))
            if 'armature' in params:
                joint.set('armature', str(params['armature']))
        
        return mjcf_root
    
    def _add_solver_options(self, mjcf_root: ET.Element, config: Dict[str, Any]) -> ET.Element:
        """Add solver and simulation options to MJCF."""
        solver_config = config.get('solver', {})
        sim_config = config.get('simulation', {})
        
        # Find or create option element
        option = mjcf_root.find('option')
        if option is None:
            option = ET.SubElement(mjcf_root, 'option')
        
        # Set timestep
        if 'timestep' in sim_config:
            option.set('timestep', str(sim_config['timestep']))
        
        # Set gravity
        if 'gravity' in sim_config:
            gravity = sim_config['gravity']
            option.set('gravity', f"{gravity[0]} {gravity[1]} {gravity[2]}")
        
        # Set solver iterations
        if 'iterations' in solver_config:
            option.set('iterations', str(solver_config['iterations']))
        if 'tolerance' in solver_config:
            option.set('tolerance', str(solver_config['tolerance']))
        if 'solver' in solver_config:
            option.set('solver', str(solver_config['solver']))
        if 'integrator' in solver_config:
            option.set('integrator', str(solver_config['integrator']))
        if 'cone' in solver_config:
            option.set('cone', str(solver_config['cone']))
        
        return mjcf_root

    def _portable_meshdir(self) -> str:
        """meshdir written RELATIVE to the output XML's directory.

        PORTABILITY: an absolute meshdir bakes this machine's path into the
        committed XML and breaks the sim for anyone who clones or moves the
        repo (mesh STLs "not found"). A relative path resolves correctly from
        wherever the file lives. simulation.py::_resolve_meshdir converts it
        back to absolute at load time for MuJoCo's from_xml_string().
        """
        return os.path.relpath(self.meshes_path.resolve(),
                               self.output_path.resolve())

    def _fix_meshdir(self, mjcf_root: ET.Element) -> ET.Element:
        """Ensure compiler meshdir is a portable RELATIVE path to meshes."""
        compiler = mjcf_root.find('compiler')
        if compiler is None:
            compiler = ET.SubElement(mjcf_root, 'compiler')
        compiler.set('meshdir', self._portable_meshdir())
        return mjcf_root
    
    def _add_ground_plane(self, mjcf_root: ET.Element, config: Dict[str, Any]) -> ET.Element:
        """Add a ground plane for the robot to interact with."""
        worldbody = mjcf_root.find('.//worldbody')
        if worldbody is None:
            return mjcf_root
        
        # Check if ground already exists
        for geom in worldbody.findall('geom'):
            if geom.get('name') == 'ground':
                return mjcf_root
        
        # Get config
        ground_config = config.get('ground', {})
        pos = ground_config.get('position', [0, 0, -0.5])
        friction = ground_config.get('friction', [3, 0.5, 0.1])
        rgba = ground_config.get('rgba', [0.5, 0.5, 0.5, 1])
        condim = ground_config.get('condim')
        solref = ground_config.get('solref')
        solimp = ground_config.get('solimp')
        
        # Add ground plane
        ground = ET.Element('geom')
        ground.set('name', 'ground')
        ground.set('type', 'plane')
        ground.set('size', '10 10 0.1')
        ground.set('rgba', f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}')
        ground.set('pos', f'{pos[0]} {pos[1]} {pos[2]}')
        ground.set('friction', f'{friction[0]} {friction[1]} {friction[2]}')
        if condim is not None:
            ground.set('condim', str(condim))
        if solref is not None:
            ground.set('solref', f'{solref[0]} {solref[1]}')
        if solimp is not None:
            ground.set('solimp', f'{solimp[0]} {solimp[1]} {solimp[2]}')
        
        # Insert ground at beginning of worldbody
        worldbody.insert(0, ground)
        
        return mjcf_root
    
    def _set_initial_position(self, mjcf_root: ET.Element, config: Dict[str, Any]) -> ET.Element:
        """Set initial robot position from config."""
        sim_config = config.get('simulation', {})
        initial_height = sim_config.get('initial_height', 0.3)
        
        worldbody = mjcf_root.find('.//worldbody')
        if worldbody is None:
            return mjcf_root
        
        # Find base_link body
        base_body = worldbody.find('body')
        if base_body is not None:
            base_body.set('pos', f'0 0 {initial_height}')
        
        return mjcf_root
    
    def _apply_outer_shell_friction(self, mjcf_root: ET.Element, config: Dict[str, Any]) -> ET.Element:
        """Apply friction to outer shell geometry."""
        outer_config = config.get('outer_shell', {})
        contact_cfg = config.get('contact', {})
        outer_body = outer_config.get('body') or contact_cfg.get('outer_body', 'Link3')
        friction = outer_config.get('friction', [5, 1, 0.5])
        condim = outer_config.get('condim')
        solref = outer_config.get('solref')
        solimp = outer_config.get('solimp')
        
        # Find outer shell geom
        for body in mjcf_root.findall(f'.//body[@name="{outer_body}"]'):
            for geom in body.findall('geom'):
                geom.set('friction', f'{friction[0]} {friction[1]} {friction[2]}')
                if condim is not None:
                    geom.set('condim', str(condim))
                if solref is not None:
                    geom.set('solref', f'{solref[0]} {solref[1]}')
                if solimp is not None:
                    geom.set('solimp', f'{solimp[0]} {solimp[1]} {solimp[2]}')
        
        return mjcf_root

    def _apply_contact_filters(self, mjcf_root: ET.Element, config: Dict[str, Any]) -> ET.Element:
        """Disable internal self-collisions; keep outer shell + ground collidable."""
        contact_cfg = config.get('contact', {})
        internal_bodies = contact_cfg.get('internal_bodies', ['base_link', 'Link1', 'Link2'])
        outer_body = contact_cfg.get('outer_body', 'Link3')

        # Disable collisions for internal bodies
        for body_name in internal_bodies:
            for body in mjcf_root.findall(f'.//body[@name="{body_name}"]'):
                for geom in body.findall('geom'):
                    geom.set('contype', '0')
                    geom.set('conaffinity', '0')

        # Ensure outer shell collides
        for body in mjcf_root.findall(f'.//body[@name="{outer_body}"]'):
            for geom in body.findall('geom'):
                geom.set('contype', '1')
                geom.set('conaffinity', '1')

        # Ensure ground collides
        for geom in mjcf_root.findall('.//worldbody/geom[@name="ground"]'):
            geom.set('contype', '1')
            geom.set('conaffinity', '1')

        return mjcf_root


    def _add_ballast(self, mjcf_root: ET.Element, config: Dict[str, Any]) -> ET.Element:
        """Add a small internal ballast mass to create inertial coupling."""
        ballast_cfg = config.get('ballast', {})
        if not ballast_cfg.get('enabled', False):
            return mjcf_root

        base_body = mjcf_root.find('.//worldbody/body[@name="base_link"]')
        if base_body is None:
            return mjcf_root

        pos = ballast_cfg.get('pos', [0.05, 0.0, 0.0])
        mass = ballast_cfg.get('mass', 0.18)
        radius = ballast_cfg.get('radius', 0.02)
        rgba = ballast_cfg.get('rgba', [0.9, 0.2, 0.2, 0.6])

        ballast_body = ET.SubElement(base_body, 'body')
        ballast_body.set('name', 'ballast')
        ballast_body.set('pos', f'{pos[0]} {pos[1]} {pos[2]}')

        geom = ET.SubElement(ballast_body, 'geom')
        geom.set('type', 'sphere')
        geom.set('size', f'{radius}')
        geom.set('mass', f'{mass}')
        geom.set('rgba', f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}')
        geom.set('contype', '0')
        geom.set('conaffinity', '0')

        return mjcf_root

    def _apply_visual_colors(self, mjcf_root: ET.Element, config: Dict[str, Any]) -> ET.Element:
        """Apply per-link RGBA colors from config."""
        visual_cfg = config.get('visual', {})
        if not visual_cfg:
            return mjcf_root

        for body in mjcf_root.findall('.//body'):
            name = body.get('name')
            if name and name in visual_cfg:
                rgba = visual_cfg[name]
                for geom in body.findall('geom'):
                    geom.set('rgba', f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}')

        # Optional ground color (if already created)
        ground_cfg = config.get('ground', {})
        ground_rgba = ground_cfg.get('rgba')
        if ground_rgba is not None:
            for geom in mjcf_root.findall('.//worldbody/geom[@name="ground"]'):
                geom.set('rgba', f'{ground_rgba[0]} {ground_rgba[1]} {ground_rgba[2]} {ground_rgba[3]}')

        return mjcf_root
    
    def convert_to_mjcf(self, add_ground: bool = True) -> str:
        """
        Convert URDF to MJCF with freejoint and physics parameters.
        
        Args:
            add_ground: Whether to add a ground plane
            
        Returns:
            Path to the generated MJCF file
        """
        import mujoco
        
        # Read and fix URDF
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        
        with open(self.urdf_path, 'r', encoding='utf-8') as f:
            urdf_content = f.read()
        
        # Fix mesh paths
        urdf_content = self._fix_mesh_paths(urdf_content)
        
        # Save temporary URDF with fixed paths
        temp_urdf = self.output_path / "temp_robot.urdf"
        with open(temp_urdf, 'w', encoding='utf-8') as f:
            f.write(urdf_content)
        
        # Load into MuJoCo to get MJCF
        try:
            model = mujoco.MjModel.from_xml_path(str(temp_urdf))
            
            # Export to MJCF XML
            mjcf_content = mujoco.mj_saveLastXML(str(self.output_path / "temp_export.xml"), model)
            
            # Read the exported MJCF
            with open(self.output_path / "temp_export.xml", 'r') as f:
                mjcf_content = f.read()
                
        except Exception as e:
            # If direct conversion fails, manually create MJCF from URDF
            print(f"Direct conversion failed ({e}), using manual conversion...")
            mjcf_content = self._manual_urdf_to_mjcf(urdf_content)
        
        # Parse MJCF
        mjcf_root = ET.fromstring(mjcf_content)
        mjcf_root.set('model', self.model_name)
        
        # Load config
        config = self.load_config()
        
        # Apply modifications
        mjcf_root = self._add_freejoint_to_mjcf(mjcf_root)
        mjcf_root = self._inject_physics_params(mjcf_root, config)
        mjcf_root = self._add_solver_options(mjcf_root, config)
        mjcf_root = self._fix_meshdir(mjcf_root)
        
        if add_ground:
            mjcf_root = self._add_ground_plane(mjcf_root, config)
        
        # Apply initial position and outer shell friction
        mjcf_root = self._set_initial_position(mjcf_root, config)
        mjcf_root = self._apply_outer_shell_friction(mjcf_root, config)
        mjcf_root = self._add_ballast(mjcf_root, config)
        mjcf_root = self._apply_contact_filters(mjcf_root, config)
        mjcf_root = self._apply_visual_colors(mjcf_root, config)
        # Save final MJCF
        output_file = self.output_path / self.output_name
        tree = ET.ElementTree(mjcf_root)
        ET.indent(tree, space="  ")
        tree.write(output_file, encoding='unicode', xml_declaration=True)
        
        # Clean up temp files
        for temp_file in ['temp_robot.urdf', 'temp_export.xml']:
            temp_path = self.output_path / temp_file
            if temp_path.exists():
                temp_path.unlink()
        
        print(f"✓ Generated MJCF: {output_file}")
        return str(output_file)
    
    def _manual_urdf_to_mjcf(self, urdf_content: str) -> str:
        """
        Manually convert URDF to MJCF format.
        Falls back to this if MuJoCo's built-in converter fails.
        """
        urdf_root = ET.fromstring(urdf_content)
        
        # Create MJCF structure
        mujoco_root = ET.Element('mujoco')
        mujoco_root.set('model', self.model_name)
        
        # Add compiler settings
        compiler = ET.SubElement(mujoco_root, 'compiler')
        compiler.set('angle', 'radian')
        # PORTABILITY: relative meshdir — see _portable_meshdir().
        compiler.set('meshdir', self._portable_meshdir())
        
        # Add assets (meshes)
        asset = ET.SubElement(mujoco_root, 'asset')
        for link in urdf_root.findall('.//link'):
            mesh_elem = link.find('.//mesh')
            if mesh_elem is not None:
                filename = mesh_elem.get('filename', '')
                # Extract mesh name from path
                mesh_name = Path(filename).stem if filename else link.get('name', '')
                mesh_file = Path(filename).name if filename else f"{mesh_name}.STL"
                
                mesh_asset = ET.SubElement(asset, 'mesh')
                mesh_asset.set('name', mesh_name)
                mesh_asset.set('file', mesh_file)
        
        # Create worldbody
        worldbody = ET.SubElement(mujoco_root, 'worldbody')
        
        # Build body hierarchy from URDF
        self._build_body_hierarchy(urdf_root, worldbody)
        
        return ET.tostring(mujoco_root, encoding='unicode')
    
    def _build_body_hierarchy(self, urdf_root: ET.Element, parent: ET.Element,
                              parent_link_name: str = None):
        """Recursively build MJCF body hierarchy from URDF."""
        # Find base link (link with no parent joint)
        child_links = {j.find('child').get('link') for j in urdf_root.findall('.//joint')}
        
        if parent_link_name is None:
            # Find root link
            for link in urdf_root.findall('.//link'):
                link_name = link.get('name')
                if link_name not in child_links:
                    self._add_body_from_link(urdf_root, parent, link)
                    break
        else:
            # Find joints with this parent
            for joint in urdf_root.findall('.//joint'):
                parent_elem = joint.find('parent')
                if parent_elem is not None and parent_elem.get('link') == parent_link_name:
                    child_link_name = joint.find('child').get('link')
                    child_link = urdf_root.find(f".//link[@name='{child_link_name}']")
                    if child_link is not None:
                        self._add_body_from_link(urdf_root, parent, child_link, joint)
    
    def _add_body_from_link(self, urdf_root: ET.Element, parent: ET.Element,
                            link: ET.Element, joint: ET.Element = None):
        """Add a MJCF body element from URDF link."""
        link_name = link.get('name')
        body = ET.SubElement(parent, 'body')
        body.set('name', link_name)
        
        # Set position from joint origin
        if joint is not None:
            origin = joint.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0')
                rpy = origin.get('rpy', '0 0 0')
                body.set('pos', xyz)
                # Convert RPY to quaternion would be ideal, but for simplicity use euler
                body.set('euler', rpy)
            
            # Add joint
            joint_elem = ET.SubElement(body, 'joint')
            joint_elem.set('name', joint.get('name'))
            
            joint_type = joint.get('type', 'revolute')
            if joint_type == 'continuous':
                joint_elem.set('type', 'hinge')
                joint_elem.set('limited', 'false')
            elif joint_type == 'revolute':
                joint_elem.set('type', 'hinge')
                limit = joint.find('limit')
                if limit is not None:
                    joint_elem.set('range', f"{limit.get('lower', '-3.14')} {limit.get('upper', '3.14')}")
            
            axis = joint.find('axis')
            if axis is not None:
                joint_elem.set('axis', axis.get('xyz', '0 0 1'))
        
        # Add inertial
        inertial = link.find('inertial')
        if inertial is not None:
            inertial_elem = ET.SubElement(body, 'inertial')
            origin = inertial.find('origin')
            if origin is not None:
                inertial_elem.set('pos', origin.get('xyz', '0 0 0'))
            mass = inertial.find('mass')
            if mass is not None:
                inertial_elem.set('mass', mass.get('value', '1'))
            inertia = inertial.find('inertia')
            if inertia is not None:
                # Convert to MuJoCo inertia format (diagonal)
                ixx = inertia.get('ixx', '0.01')
                iyy = inertia.get('iyy', '0.01')
                izz = inertia.get('izz', '0.01')
                inertial_elem.set('diaginertia', f"{ixx} {iyy} {izz}")
        
        # Add visual geometry
        visual = link.find('visual')
        if visual is not None:
            geom = ET.SubElement(body, 'geom')
            geom.set('type', 'mesh')
            mesh_name = link_name.replace(' ', '_')
            geom.set('mesh', mesh_name)
            geom.set('rgba', '0.8 0.8 0.8 1')
        
        # Recursively add child bodies
        self._build_body_hierarchy(urdf_root, body, link_name)


def main():
    """CLI entrypoint for MJCF generation."""
    parser = argparse.ArgumentParser(description="URDF -> MJCF converter")
    parser.add_argument('--robot-path', type=str, default=None,
                        help='Robot folder path (default: simulation/robot_v1)')
    parser.add_argument('--urdf', type=str, default=None,
                        help='URDF filename inside robot-path/urdf')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config path (default: robot_params.yaml)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output MJCF filename (default: <model>_mujoco.xml)')
    parser.add_argument('--model-name', type=str, default=None,
                        help='MJCF model name (default: robot folder name)')
    parser.add_argument('--no-ground', action='store_true',
                        help='Skip adding ground plane')
    args = parser.parse_args()

    loader = RobotLoader(
        robot_path=args.robot_path,
        urdf_filename=args.urdf,
        output_name=args.output,
        config_path=args.config,
        model_name=args.model_name,
    )
    mjcf_path = loader.convert_to_mjcf(add_ground=not args.no_ground)
    print(f"Generated: {mjcf_path}")


if __name__ == "__main__":
    main()
