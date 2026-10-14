//! Standalone line-of-sight executable. Polygon segment predicates derive
//! from polygon_coverage/rust/src/main.rs; legacy Bresenham is retained.
//! New paths implement conservative DDA and analytic 3D CSG interval clipping.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{env, fs::File, io::{BufReader, BufWriter, Write}, time::Instant};

#[derive(Deserialize)]
struct Primitive {
    kind: String,
    center: Vec<f64>,
    #[serde(default)] radii: Vec<f64>,
    #[serde(default)] planes: Vec<[f64; 4]>,
}
#[derive(Deserialize)]
struct Solid { positive: Vec<Primitive>, negative: Vec<Primitive> }
#[derive(Deserialize)]
struct Grid { shape: Vec<usize>, spacing: Vec<f64>, origin: Vec<f64>, free: Vec<bool> }
#[derive(Deserialize, Default)]
struct Sensor {
    range: Option<f64>,
    axis: Option<Vec<f64>>,
    half_angle_degrees: Option<f64>,
    #[serde(default)] boundary_targets: bool,
}
#[derive(Deserialize)]
struct Input {
    schema_version: usize, dimension: usize, mode: String,
    observers: Vec<Vec<f64>>, targets: Vec<Vec<f64>>,
    #[serde(default)] polygons: Vec<Vec<[f64;2]>>,
    solid: Option<Solid>, grid: Option<Grid>,
    #[serde(default)] sensor: Sensor,
    epsilon: Option<f64>,
}
#[derive(Serialize)]
struct Output {
    schema_version: usize, dimension: usize, mode: String,
    observer_count: usize, target_count: usize, nonzero: usize,
    per_observer_counts: Vec<usize>, computation_seconds: f64, threads: usize,
    matrix_format: String, numerical_epsilon: f64,
}

fn cross(a: [f64;2], b: [f64;2]) -> f64 { a[0]*b[1]-a[1]*b[0] }
fn sub(a: [f64;2], b: [f64;2]) -> [f64;2] { [a[0]-b[0], a[1]-b[1]] }

// Reused geometric convention: intermediate boundary contact blocks;
// terminal contact can be explicitly enabled for boundary targets.
fn edge_blocked(a: [f64;2], b: [f64;2], p: [f64;2], q: [f64;2]) -> bool {
    let delta = sub(q,p); let segment = sub(b,a); let offset = sub(a,p);
    let denominator = cross(delta, segment);
    let tolerance = 1e-12*((delta[0].abs()+delta[1].abs())*
                           (segment[0].abs()+segment[1].abs())).max(1.);
    if denominator.abs() <= tolerance {
        if cross(offset,delta).abs() > tolerance { return false; }
        let axis = if delta[0].abs() >= delta[1].abs() {0} else {1};
        if delta[axis].abs() < 1e-14 { return false; }
        let t0 = (a[axis]-p[axis])/delta[axis]; let t1 = (b[axis]-p[axis])/delta[axis];
        return t0.min(t1).max(1e-10) < t0.max(t1).min(1.-1e-10);
    }
    let t = cross(offset,segment)/denominator; let u = cross(offset,delta)/denominator;
    t > 1e-10 && t < 1.-1e-10 && u >= -1e-10 && u <= 1.+1e-10
}

fn point_in_polygon(rings: &[Vec<[f64;2]>], p: &[f64], boundary: bool) -> bool {
    let mut inside = false;
    for ring in rings {
        for i in 0..ring.len() {
            let a=ring[i]; let b=ring[(i+1)%ring.len()];
            let edge=sub(b,a); let offset=[p[0]-a[0],p[1]-a[1]];
            let norm=(edge[0]*edge[0]+edge[1]*edge[1]).sqrt();
            if norm>0. && cross(edge,offset).abs() < 1e-9*norm &&
                p[0]>=a[0].min(b[0])-1e-9 && p[0]<=a[0].max(b[0])+1e-9 &&
                p[1]>=a[1].min(b[1])-1e-9 && p[1]<=a[1].max(b[1])+1e-9 { return boundary; }
            if (a[1]>p[1]) != (b[1]>p[1]) && p[0] < (b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0] {
                inside=!inside;
            }
        }
    }
    inside
}

fn primitive_inside(primitive: &Primitive, p: &[f64]) -> bool {
    if primitive.kind=="polyhedron" {
        primitive.planes.iter().all(|plane| (0..3).map(|i|plane[i]*p[i]).sum::<f64>()+plane[3] < 0.)
    } else {
        (0..3).map(|i| ((p[i]-primitive.center[i])/primitive.radii[i]).powi(2)).sum::<f64>() < 1.
    }
}
fn solid_inside(s: &Solid, p: &[f64]) -> bool {
    s.positive.iter().any(|r|primitive_inside(r,p)) && !s.negative.iter().any(|r| {
        if r.kind=="polyhedron" {
            r.planes.iter().all(|plane| (0..3).map(|i|plane[i]*p[i]).sum::<f64>()+plane[3] <= 1e-12)
        } else {
            (0..3).map(|i| ((p[i]-r.center[i])/r.radii[i]).powi(2)).sum::<f64>() <= 1.+1e-12
        }
    })
}
fn primitive_interval(primitive: &Primitive, p: &[f64], q: &[f64], margin: f64) -> Option<(f64,f64)> {
    let mut lo: f64=0.; let mut hi: f64=1.;
    if primitive.kind=="polyhedron" {
        for plane in &primitive.planes {
            let norm=(plane[0]*plane[0]+plane[1]*plane[1]+plane[2]*plane[2]).sqrt();
            let value=(0..3).map(|i|plane[i]*p[i]).sum::<f64>()+plane[3]+margin*norm;
            let slope=(0..3).map(|i|plane[i]*(q[i]-p[i])).sum::<f64>();
            if slope.abs()<1e-13 {
                if value>=0. {return None;}
            } else if slope>0. {hi=hi.min(-value/slope);}
            else {lo=lo.max(-value/slope);}
            if hi<lo {return None;}
        }
    } else {
        let mut a=0.; let mut b=0.; let mut c=-1.;
        for i in 0..3 {
            let radius=primitive.radii[i]-margin;
            if radius<=0. {return None;}
            let source=(p[i]-primitive.center[i])/radius; let delta=(q[i]-p[i])/radius;
            a+=delta*delta; b+=2.*source*delta; c+=source*source;
        }
        if a<1e-24 {return if c<0. {Some((0.,1.))} else {None};}
        let discriminant=b*b-4.*a*c;
        if discriminant<0. {return None;}
        lo=lo.max((-b-discriminant.sqrt())/(2.*a));
        hi=hi.min((-b+discriminant.sqrt())/(2.*a));
    }
    if hi>=lo {Some((lo,hi))} else {None}
}

fn solid_visible(s: &Solid, p: &[f64], q: &[f64], epsilon: f64) -> bool {
    let mut intervals: Vec<_>=s.positive.iter().filter_map(|r|primitive_interval(r,p,q,epsilon)).collect();
    intervals.sort_by(|a,b|a.0.total_cmp(&b.0));
    let mut reached: f64=0.;
    for (lo,hi) in intervals {
        if lo>reached+1e-10 {return false;}
        reached=reached.max(hi);
    }
    if reached<1.-1e-10 {return false;}
    !s.negative.iter().any(|r|primitive_interval(r,p,q,-epsilon).is_some())
}

fn grid_index(grid: &Grid, cell: &[isize]) -> Option<usize> {
    let mut index=0;
    for (i,&coordinate) in cell.iter().enumerate() {
        if coordinate<0 || coordinate>=grid.shape[i] as isize {return None;}
        index=index*grid.shape[i]+coordinate as usize;
    }
    Some(index)
}
fn grid_free(grid: &Grid, cell: &[isize]) -> bool {
    grid_index(grid,cell).map(|i|grid.free[i]).unwrap_or(false)
}
fn cell_of(grid: &Grid, point: &[f64]) -> Vec<isize> {
    point.iter().enumerate().map(|(i,&v)| ((v-grid.origin[i])/grid.spacing[i]).floor() as isize).collect()
}
fn grid_inside(grid: &Grid, point: &[f64]) -> bool {grid_free(grid,&cell_of(grid,point))}

fn dda_visible(grid: &Grid, p: &[f64], q: &[f64]) -> bool {
    let dimension=grid.shape.len();
    let source: Vec<_>=(0..dimension).map(|i|(p[i]-grid.origin[i])/grid.spacing[i]).collect();
    let delta: Vec<_>=(0..dimension).map(|i|(q[i]-p[i])/grid.spacing[i]).collect();
    let mut cell: Vec<_>=source.iter().map(|x|x.floor() as isize).collect();
    let step: Vec<_>=delta.iter().map(|x|if *x>=0. {1} else {-1}).collect();
    let dt: Vec<_>=delta.iter().map(|x|if x.abs()<1e-15 {f64::INFINITY} else {1./x.abs()}).collect();
    let mut tmax: Vec<_>=(0..dimension).map(|i|if delta[i].abs()<1e-15 {f64::INFINITY}
        else {(cell[i] as f64+if step[i]>0 {1.} else {0.}-source[i])/delta[i]}).collect();
    // A ray lying on a cell face checks both sides of that face.
    let fixed_faces: Vec<_>=(0..dimension).filter(|&i|delta[i].abs()<1e-15 &&
                          (source[i]-source[i].round()).abs()<1e-10).collect();
    let check=|base: &[isize]| -> bool {
        for mask in 0..(1<<fixed_faces.len()) {
            let mut other=base.to_vec();
            for (bit,&axis) in fixed_faces.iter().enumerate() {
                if mask & (1<<bit)!=0 {other[axis]-=1;}
            }
            if !grid_free(grid,&other) {return false;}
        }
        true
    };
    for _ in 0..(grid.shape.iter().sum::<usize>()*3+16) {
        if !check(&cell) {return false;}
        let next=tmax.iter().copied().fold(f64::INFINITY,f64::min);
        if next>=1.-1e-12 {return true;}
        let axes: Vec<_>=(0..dimension).filter(|&i|(tmax[i]-next).abs()<1e-10).collect();
        // At an edge/corner every side-adjacent cell is visited: supercover.
        for mask in 1..(1<<axes.len()) {
            let mut neighbor=cell.clone();
            for (bit,&axis) in axes.iter().enumerate() {
                if mask & (1<<bit)!=0 {neighbor[axis]+=step[axis];}
            }
            if !check(&neighbor) {return false;}
        }
        for axis in axes {cell[axis]+=step[axis]; tmax[axis]+=dt[axis];}
    }
    false
}

fn bresenham_direction(grid: &Grid, start: &[isize], end: &[isize]) -> bool {
    let (mut x,mut y)=(start[0],start[1]); let (x1,y1)=(end[0],end[1]);
    let dx=(x1-x).abs(); let dy=-(y1-y).abs();
    let sx=if x<x1 {1} else {-1}; let sy=if y<y1 {1} else {-1};
    let mut error=dx+dy;
    loop {
        if !grid_free(grid,&[x,y]) {return false;}
        if x==x1 && y==y1 {return true;}
        let twice=2*error;
        if twice>=dy {error+=dy; x+=sx;}
        if twice<=dx {error+=dx; y+=sy;}
    }
}
fn bresenham_visible(grid: &Grid,p: &[f64],q: &[f64]) -> bool {
    let a=cell_of(grid,p); let b=cell_of(grid,q);
    // Strict-interior, omnidirectional symmetry: both directional rasters
    // must be clear. This legacy mode is not the default corner-safe mode.
    bresenham_direction(grid,&a,&b) && bresenham_direction(grid,&b,&a)
}
fn sensor_accepts(input: &Input,p: &[f64],q: &[f64]) -> bool {
    let delta: Vec<_>=p.iter().zip(q).map(|(a,b)|b-a).collect();
    let length=delta.iter().map(|x|x*x).sum::<f64>().sqrt();
    if input.sensor.range.map(|r|length>r).unwrap_or(false) {return false;}
    if length<1e-14 {return true;}
    if let (Some(axis),Some(angle))=(&input.sensor.axis,input.sensor.half_angle_degrees) {
        let norm=axis.iter().map(|x|x*x).sum::<f64>().sqrt();
        let cosine=axis.iter().zip(&delta).map(|(a,b)|a*b).sum::<f64>()/(norm*length);
        if cosine+1e-12 < angle.to_radians().cos() {return false;}
    }
    true
}
fn visible(input: &Input,p: &[f64],q: &[f64]) -> bool {
    if !sensor_accepts(input,p,q) {return false;}
    let distance=p.iter().zip(q).map(|(a,b)|(a-b).powi(2)).sum::<f64>();
    if distance<1e-24 {return true;}
    match input.mode.as_str() {
        "continuous" if input.dimension==2 => {
            let a=[p[0],p[1]]; let b=[q[0],q[1]];
            !input.polygons.iter().any(|ring| (0..ring.len()).any(|i|
                edge_blocked(ring[i],ring[(i+1)%ring.len()],a,b)))
        },
        "continuous" => solid_visible(input.solid.as_ref().unwrap(),p,q,input.epsilon.unwrap_or(1e-8)),
        "discrete" => dda_visible(input.grid.as_ref().unwrap(),p,q),
        "bresenham" => bresenham_visible(input.grid.as_ref().unwrap(),p,q),
        _ => false,
    }
}

fn validate(input: &Input) -> Result<(),String> {
    if input.schema_version!=1 || ![2,3].contains(&input.dimension) {return Err("unsupported schema/dimension".into());}
    if input.observers.is_empty() || input.targets.is_empty() {return Err("observers and targets must be nonempty".into());}
    if !["continuous","discrete","bresenham"].contains(&input.mode.as_str()) {return Err("unsupported mode".into());}
    if input.mode=="bresenham" && input.dimension!=2 {return Err("Bresenham mode supports 2D only".into());}
    if input.epsilon.map(|v|!v.is_finite() || v<=0.).unwrap_or(false) {return Err("epsilon must be finite and positive".into());}
    if let Some(range)=input.sensor.range {if !range.is_finite() || range<0. {return Err("range must be nonnegative".into());}}
    if input.sensor.axis.is_some()!=input.sensor.half_angle_degrees.is_some() {return Err("axis and half angle must be supplied together".into());}
    if let Some(axis)=&input.sensor.axis {
        if axis.len()!=input.dimension || axis.iter().any(|v|!v.is_finite()) || axis.iter().map(|v|v*v).sum::<f64>()<=0. {
            return Err("field-of-view axis is invalid".into());
        }
    }
    if let Some(angle)=input.sensor.half_angle_degrees {if !angle.is_finite() || !(0. ..=180.).contains(&angle) {return Err("invalid half angle".into());}}
    if input.mode=="continuous" {
        if input.dimension==2 {
            if input.polygons.is_empty() || input.polygons.iter().any(|r|r.len()<3 || r.iter().flatten().any(|v|!v.is_finite())) {
                return Err("invalid polygon rings".into());
            }
        } else {
            let s=input.solid.as_ref().ok_or("continuous 3D needs a solid")?;
            if s.positive.is_empty() {return Err("empty solid".into());}
            for primitive in s.positive.iter().chain(&s.negative) {
                if primitive.center.len()!=3 || primitive.center.iter().any(|v|!v.is_finite()) {return Err("invalid primitive center".into());}
                if primitive.kind=="polyhedron" {
                    if primitive.planes.len()<4 || primitive.planes.iter().any(|p|
                        p.iter().any(|v|!v.is_finite()) || p[..3].iter().map(|v|v*v).sum::<f64>()<1e-20) {return Err("invalid polyhedron planes".into());}
                } else if primitive.kind=="ellipsoid" {
                    if primitive.radii.len()!=3 || primitive.radii.iter().any(|v|!v.is_finite() || *v<=0.) {return Err("invalid ellipsoid radii".into());}
                } else {return Err("unknown solid primitive".into());}
            }
        }
    } else {
        let g=input.grid.as_ref().ok_or("discrete mode needs a grid")?;
        if g.shape.len()!=input.dimension || g.spacing.len()!=input.dimension || g.origin.len()!=input.dimension ||
            g.shape.iter().any(|&n|n==0) || g.spacing.iter().any(|v|!v.is_finite() || *v<=0.) ||
            g.origin.iter().any(|v|!v.is_finite()) || g.shape.iter().product::<usize>()!=g.free.len() {return Err("invalid grid shape/spacing/data".into());}
    }
    for (is_target,points) in [(false,&input.observers),(true,&input.targets)] {
        for (i,p) in points.iter().enumerate() {
            if p.len()!=input.dimension || p.iter().any(|v|!v.is_finite()) {return Err(format!("invalid point {i}"));}
            let inside=if input.mode=="continuous" && input.dimension==2 {
                point_in_polygon(&input.polygons,p,is_target && input.sensor.boundary_targets)
            } else if input.mode=="continuous" {solid_inside(input.solid.as_ref().unwrap(),p)}
            else {grid_inside(input.grid.as_ref().unwrap(),p)};
            if !inside {return Err(format!("{} {i} is not in free space",if is_target {"target"} else {"observer"}));}
        }
    }
    Ok(())
}

fn run() -> Result<(),String> {
    let args: Vec<String>=env::args().collect();
    if args.len()==1 || args.iter().any(|s|s=="--help" || s=="-h") {
        println!("visibility_engine --input input.json --output PREFIX [--threads N]\n\
Schema 1 modes: continuous (2D polygon / 3D CSG), discrete (2D/3D conservative DDA), bresenham (2D legacy).\n\
Outputs PREFIX.csr (little-endian WGVCSR01 offsets+indices) and PREFIX.json.\n\
Observers and targets use world coordinates. Default: omnidirectional, no range limit.\n\
See README.md and generated visibility_*_input.json files for examples.");
        return Ok(());
    }
    let mut input_path=None; let mut output_path=None; let mut threads=4;
    let mut i=1;
    while i<args.len() {
        let value=args.get(i+1).ok_or(format!("missing value after {}",args[i]))?;
        match args[i].as_str() {
            "--input"=>input_path=Some(value.clone()), "--output"=>output_path=Some(value.clone()),
            "--threads"=>threads=value.parse::<usize>().map_err(|_|"invalid thread count")?,
            other=>return Err(format!("unknown option {other}")),
        }
        i+=2;
    }
    if threads==0 {return Err("thread count must be positive".into());}
    let path=input_path.ok_or("--input is required")?;
    let prefix=output_path.ok_or("--output is required")?;
    let file=File::open(&path).map_err(|e|format!("{path}: {e}"))?;
    let input: Input=serde_json::from_reader(BufReader::new(file)).map_err(|e|format!("input JSON: {e}"))?;
    validate(&input)?;
    let pool=rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e|e.to_string())?;
    let started=Instant::now();
    let rows: Vec<Vec<usize>>=pool.install(|| input.observers.par_iter().map(|observer|
        input.targets.iter().enumerate().filter_map(|(j,target)|if visible(&input,observer,target) {Some(j)} else {None}).collect()).collect());
    let elapsed=started.elapsed().as_secs_f64();
    let counts: Vec<_>=rows.iter().map(|r|r.len()).collect();
    let nonzero=counts.iter().sum();
    let output=Output {schema_version:1,dimension:input.dimension,mode:input.mode.clone(),
        observer_count:rows.len(),target_count:input.targets.len(),nonzero,
        per_observer_counts:counts,computation_seconds:elapsed,threads,
        matrix_format:"WGVCSR01: magic[8], rows/columns/nnz u64, offsets u64, indices u64, little endian".into(),
        numerical_epsilon:input.epsilon.unwrap_or(1e-8)};
    let mut binary=BufWriter::new(File::create(format!("{prefix}.csr")).map_err(|e|e.to_string())?);
    binary.write_all(b"WGVCSR01").map_err(|e|e.to_string())?;
    for value in [rows.len(),input.targets.len(),nonzero] {binary.write_all(&(value as u64).to_le_bytes()).map_err(|e|e.to_string())?;}
    let mut offset=0;
    binary.write_all(&0u64.to_le_bytes()).map_err(|e|e.to_string())?;
    for row in &rows {offset+=row.len(); binary.write_all(&(offset as u64).to_le_bytes()).map_err(|e|e.to_string())?;}
    for row in &rows {for &j in row {binary.write_all(&(j as u64).to_le_bytes()).map_err(|e|e.to_string())?;}}
    binary.flush().map_err(|e|e.to_string())?;
    serde_json::to_writer_pretty(BufWriter::new(File::create(format!("{prefix}.json")).map_err(|e|e.to_string())?),&output).map_err(|e|e.to_string())?;
    println!("{} observers x {} targets; {} visible pairs; {:.6} s",rows.len(),input.targets.len(),nonzero,elapsed);
    Ok(())
}
fn main() {if let Err(error)=run() {eprintln!("visibility_engine: {error}"); std::process::exit(2);}}

#[cfg(test)]
mod tests {
    use super::*;
    fn grid(shape: Vec<usize>) -> Grid {
        let dimension=shape.len(); let n=shape.iter().product();
        Grid{shape,spacing:vec![1.;dimension],origin:vec![0.;dimension],free:vec![true;n]}
    }
    #[test] fn conservative_corner() {
        let mut g=grid(vec![3,3]); g.free[3]=false;
        assert!(!dda_visible(&g,&[0.5,0.5],&[2.5,2.5]));
        assert!(!dda_visible(&g,&[2.5,2.5],&[0.5,0.5]));
        assert!(bresenham_visible(&g,&[0.5,0.5],&[2.5,2.5]));
    }
    #[test] fn voxel_edge_and_face() {
        let mut g=grid(vec![3,3,3]); g.free[9]=false;
        assert!(!dda_visible(&g,&[0.5,0.5,0.5],&[2.5,2.5,2.5]));
        g.free[9]=true; assert!(dda_visible(&g,&[0.5,0.5,0.5],&[2.5,2.5,2.5]));
        g.free[0]=false; assert!(!dda_visible(&g,&[1.,0.5,0.5],&[1.,2.5,0.5]));
    }
    #[test] fn polygon_corner_terminal_and_collinear() {
        assert!(edge_blocked([1.,1.],[2.,1.],[0.,0.],[2.,2.]));
        assert!(!edge_blocked([1.,1.],[2.,1.],[0.,0.],[1.,1.]));
        assert!(edge_blocked([1.,1.],[2.,1.],[0.,1.],[3.,1.]));
    }
    #[test] fn ellipsoid_occlusion_and_tangency() {
        let ball=|center,radius|Primitive{kind:"ellipsoid".into(),center,
                                       radii:vec![radius;3],planes:vec![]};
        let s=Solid{positive:vec![ball(vec![0.,0.,0.],5.)],negative:vec![ball(vec![0.,0.,0.],1.)]};
        assert!(!solid_visible(&s,&[-2.,0.,0.],&[2.,0.,0.],1e-8));
        assert!(!solid_visible(&s,&[-2.,1.,0.],&[2.,1.,0.],1e-8));
        assert!(solid_visible(&s,&[-2.,1.5,0.],&[2.,1.5,0.],1e-8));
    }
}
