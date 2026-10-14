"""Scientific figures from saved geometry, representations and Rust results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import shapely
from shapely.geometry import Polygon, mapping
from shapely.geometry.polygon import orient
from scipy.sparse import load_npz
from skimage.measure import marching_cubes, euler_number
from .generation import load_world
from . import solid

INK="#243746"; WALL="#444a52"; TEAL="#007e8a"; ORANGE="#de7a22"
MUTED="#9ea9b1"; PALE="#eef1f3"; RED="#b5616d"; BLUE="#6c98b7"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
    "axes.titlesize":13,"axes.labelsize":11.5,"pdf.fonttype":42,
    "svg.fonttype":"none","savefig.facecolor":"white"})


def polygon_path(polygon):
    polygon=orient(polygon,sign=1.)
    vertices,codes=[],[]
    for ring in [polygon.exterior,*polygon.interiors]:
        points=list(ring.coords)
        vertices.extend(points)
        codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-2)+[MplPath.CLOSEPOLY])
    return MplPath(vertices,codes)


def draw_polygon(ax, geometry, color="white", edge=WALL, alpha=1, linewidth=.8):
    for part in shapely.get_parts(geometry):
        if part.geom_type=="Polygon" and part.area>1e-12:
            ax.add_patch(PathPatch(polygon_path(part),facecolor=color,edgecolor=edge,
                                   alpha=alpha,lw=linewidth))


def axes2(ax,world,size):
    ax.set_facecolor(PALE)
    draw_polygon(ax,world)
    # Interior holes are actual blocked islands.
    for ring in world.interiors:
        ax.add_patch(PathPatch(polygon_path(Polygon(ring)),facecolor=WALL,edgecolor=WALL,lw=.5))
    ax.set_xlim(0,size[0]); ax.set_ylim(0,size[1]); ax.set_aspect("equal")
    ax.set_xticks([0,size[0]/2,size[0]]); ax.set_yticks([0,size[1]/2,size[1]])
    ax.tick_params(labelsize=11,color=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#cbd1d6")


def numerical_visibility_polygon(world,observer):
    """Reuse the existing poster's vertex-angle ray construction.

    Numerical visibility region clipped to the authoritative polygon.
    This is not a symbolic visibility proof.
    """
    vertices=shapely.get_coordinates(world)
    origin=np.asarray(observer,float)
    angles=np.arctan2(vertices[:,1]-origin[1],vertices[:,0]-origin[0])
    angles=np.sort(np.concatenate([angles-1e-9,angles,angles+1e-9]))
    directions=np.column_stack([np.cos(angles),np.sin(angles)])
    length=max(world.bounds[2]-world.bounds[0],world.bounds[3]-world.bounds[1])*3
    rays=shapely.linestrings(np.stack([np.tile(origin,(len(angles),1)),
                                     origin+directions*length],axis=1))
    intersections=shapely.intersection(rays,world.boundary)
    coordinates,indices=shapely.get_coordinates(intersections,return_index=True)
    distances=np.full(len(angles),np.inf)
    np.minimum.at(distances,indices,np.linalg.norm(coordinates-origin,axis=1))
    if not np.all(np.isfinite(distances)):
        raise ValueError("a visibility ray has no boundary intersection")
    return shapely.make_valid(Polygon(origin+directions*distances[:,None])).intersection(world)


def visibility_regions(folder,world,observers,targets,matrix):
    destination=Path(folder)/"visibility_regions.geojson"
    if destination.exists():
        from shapely.geometry import shape
        regions=[shape(feature["geometry"]) for feature in json.loads(destination.read_text())["features"]]
    else:
        regions=[numerical_visibility_polygon(world,p) for p in observers]
        destination.write_text(json.dumps({"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"observer":i,"method":"numerical vertex-angle rays; not a symbolic proof"},
             "geometry":mapping(region)} for i,region in enumerate(regions)]},separators=(",",":"))+"\n")
    mismatches=[]
    for i,region in enumerate(regions):
        included=shapely.contains_xy(region,targets[:,0],targets[:,1])
        row=matrix.getrow(i).toarray().ravel()
        mismatches.append(int(np.count_nonzero(included!=row)))
    union=shapely.union_all(regions)
    report={"method":"vertex angles and +/-1e-9 rad; nearest boundary intersections; clipped union",
            "symbolic_proof":False,"sample_disagreements_with_rust":mismatches,
            "union_area_fraction_numerical":float(union.area/world.area)}
    (Path(folder)/"visibility_regions_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    return regions,union


def mesh_world(folder,world,metadata):
    cache=Path(folder)/"display_mesh.npz"
    if cache.exists():
        data=np.load(cache)
        return data["vertices"],data["faces"]
    size=np.asarray(metadata["world_size"],float)
    shape=np.array([72,72,44])
    axes=[np.linspace(-.5,b+.5,n) for b,n in zip(size,shape)]
    points=np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1).reshape(-1,3)
    field=solid.clearance(world,points).reshape(tuple(shape))
    spacing=np.array([(b+1)/(n-1) for b,n in zip(size,shape)])
    vertices,faces,_,_=marching_cubes(field,level=0,spacing=tuple(spacing),allow_degenerate=False)
    vertices-=.5
    np.savez_compressed(cache,vertices=vertices,faces=faces)
    edge_count={}
    for triangle in faces:
        for a,b in zip(triangle,np.roll(triangle,-1)):
            key=tuple(sorted((int(a),int(b))))
            edge_count[key]=edge_count.get(key,0)+1
    report={"method":"Lewiner marching cubes of analytic CSG clearance lower bound",
            "shape_xyz":shape.tolist(),"spacing_xyz":spacing.tolist(),
            "vertices":len(vertices),"triangles":len(faces),
            "closed_two_faces_per_edge":all(n==2 for n in edge_count.values()),
            "mesh_is_display_approximation":True,
            "global_surface_error_bound_certified":False,
            "warning":"individual-primitive distance bound is exact in sign; internal seams can be numerically sharp"}
    (Path(folder)/"display_mesh.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=[f"v {x:.8f} {y:.8f} {z:.8f}" for x,y,z in vertices]
    lines.extend("f "+" ".join(str(int(v)+1) for v in triangle) for triangle in faces)
    (Path(folder)/"display_mesh.obj").write_text("\n".join(lines)+"\n")
    return vertices,faces


def axes3(ax,size):
    ax.set_xlim(0,size[0]);ax.set_ylim(0,size[1]);ax.set_zlim(0,size[2])
    ax.set_box_aspect(size)
    ax.view_init(elev=24,azim=-58)
    ax.set_xticks([0,size[0]]);ax.set_yticks([0,size[1]]);ax.set_zticks([0,size[2]])
    ax.set_xlabel("x",labelpad=1);ax.set_ylabel("y",labelpad=1);ax.set_zlabel("z",labelpad=1)
    ax.tick_params(labelsize=10.5,pad=0)
    ax.grid(False)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False
        axis.line.set_color("#bdc5cb")


def draw_mesh(ax,vertices,faces,color=BLUE,alpha=.18,cut=None):
    triangles=vertices[faces]
    if cut is not None:
        triangles=triangles[np.all(triangles[:,:,2]<cut,axis=1)]
    collection=Poly3DCollection(triangles,facecolors=color,edgecolors="none",alpha=alpha,
                                linewidths=0,rasterized=True)
    ax.add_collection3d(collection)


def obstacle_meshes(ax,world):
    for primitive in world["negative"]:
        if primitive["kind"]=="polyhedron":
            triangles=np.asarray(primitive["vertices"])[np.asarray(primitive["faces"])]
            ax.add_collection3d(Poly3DCollection(triangles,facecolor=WALL,edgecolors="none",alpha=.95))
        else:
            u,v=np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi,12))
            c=np.asarray(primitive["center"]);r=np.asarray(primitive["radii"])
            ax.plot_surface(c[0]+r[0]*np.cos(u)*np.sin(v),c[1]+r[1]*np.sin(u)*np.sin(v),
                            c[2]+r[2]*np.cos(v),color=WALL,alpha=.95,linewidth=0,rasterized=True)


def draw_points3(ax,points,visible=None,observers=None):
    if visible is None:
        ax.scatter(*points.T,s=2.8,c=TEAL,alpha=.45,depthshade=False,rasterized=True)
    else:
        ax.scatter(*points[~visible].T,s=2.8,c=MUTED,alpha=.2,depthshade=False,rasterized=True)
        ax.scatter(*points[visible].T,s=4.,c=TEAL,alpha=.75,depthshade=False,rasterized=True)
    if observers is not None:
        ax.scatter(*observers.T,s=38,c=ORANGE,edgecolors="white",linewidths=.6,depthshade=False,zorder=20)


def legend(fig):
    fig.legend(handles=[Line2D([],[],marker="o",ls="",color=ORANGE,label="Observer"),
                        Line2D([],[],marker="o",ls="",color=TEAL,label="Visible target / region"),
                        Line2D([],[],marker="o",ls="",color=MUTED,label="Occluded target"),
                        Line2D([],[],marker="s",ls="",color=WALL,label="Obstacle")],
               loc="lower center",ncol=4,frameon=False,fontsize=11,bbox_to_anchor=(.5,.005))


def save_figure(fig,stem,svg=True):
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",pad_inches=.08)
    fig.savefig(stem.with_suffix(".png"),dpi=260,bbox_inches="tight",pad_inches=.08)
    if svg:
        fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight",pad_inches=.08)
    plt.close(fig)


def render_world(folder,output):
    folder=Path(folder);output=Path(output)
    world,metadata=load_world(folder)
    targets=np.load(folder/"targets.npy");observers=np.load(folder/"observers.npy")
    matrix=load_npz(folder/"visibility_continuous.npz")
    single=matrix.getrow(0).toarray().ravel().astype(bool)
    union=np.asarray(matrix.sum(axis=0)).ravel()>0
    size=metadata["world_size"]
    titles=["A  Continuous geometry","B  Conservative occupancy","C  Continuous targets",
            f"D  One observer · {single.mean():.1%} targets",
            f"E  Eight observers · {union.mean():.1%} targets","F  Unobserved targets / cutaway"]
    fig=plt.figure(figsize=(12,8.6),facecolor="white")
    if metadata["dimension"]==2:
        axes=[fig.add_subplot(2,3,i+1) for i in range(6)]
        regions,region_union=visibility_regions(folder,world,observers,targets,matrix)
        for ax in axes:
            axes2(ax,world,size)
        free=np.load(folder/"free.npy")
        axes[1].cla()
        axes[1].imshow(free.T,origin="lower",extent=[0,size[0],0,size[1]],
                       cmap=matplotlib.colors.ListedColormap([PALE,"white"]),interpolation="nearest")
        axes[1].set_aspect("equal")
        axes[1].tick_params(labelsize=11)
        axes[2].scatter(*targets.T,s=2.2,color=TEAL,alpha=.45,rasterized=True)
        draw_polygon(axes[3],regions[0],TEAL,edge="none",alpha=.18)
        draw_polygon(axes[4],region_union,TEAL,edge="none",alpha=.16)
        for ax,visible,viewpoints in [(axes[3],single,observers[:1]),(axes[4],union,observers)]:
            ax.scatter(*targets[~visible].T,s=2.,c=MUTED,alpha=.55,rasterized=True)
            ax.scatter(*targets[visible].T,s=2.5,c=TEAL,alpha=.65,rasterized=True)
            ax.scatter(*viewpoints.T,s=58,color=ORANGE,edgecolor="white",linewidth=1.2,zorder=10)
        draw_polygon(axes[5],world.difference(region_union),RED,edge="none",alpha=.5)
        axes[5].scatter(*targets[~union].T,s=5.,color=RED,rasterized=True)
        axes[5].scatter(*observers.T,s=48,color=ORANGE,edgecolor="white",linewidth=1,zorder=10)
        titles[5]="F  Unobserved region (numerical)"
    else:
        axes=[fig.add_subplot(2,3,i+1,projection="3d") for i in range(6)]
        vertices,faces=mesh_world(folder,world,metadata)
        for ax in axes:
            axes3(ax,size)
        draw_mesh(axes[0],vertices,faces,alpha=.23)
        obstacle_meshes(axes[0],world)
        for path in metadata["paths3"]:
            points=np.asarray(path["points"])
            axes[0].plot(*points.T,color=BLUE,lw=1.3,alpha=.65)
        # Display the surface extracted from the actual conservative voxel
        # array; marching-cubes interpolation is explicitly a display step.
        free=np.load(folder/"free.npy")
        spacing=np.asarray(metadata["occupancy"]["spacing_xyz"])
        vv,ff,_,_=marching_cubes(np.pad(free,1).astype(float),level=.5,
                                  spacing=tuple(spacing),allow_degenerate=False)
        vv-=spacing/2
        draw_mesh(axes[1],vv,ff,TEAL,alpha=.3)
        draw_points3(axes[2],targets)
        for ax,visible,viewpoints in [(axes[3],single,observers[:1]),(axes[4],union,observers)]:
            draw_mesh(ax,vertices,faces,alpha=.04)
            draw_points3(ax,targets,visible,viewpoints)
        # Selected computed visible segments demonstrate depth, without
        # inventing any visibility cone or volumetric coverage claim.
        for target in targets[single][::max(1,int(single.sum()/12))]:
            axes[3].plot(*np.vstack([observers[0],target]).T,color=TEAL,lw=.55,alpha=.5)
        cut=size[2]*.56
        draw_mesh(axes[5],vertices,faces,alpha=.32,cut=cut)
        lower=targets[:,2]<cut
        draw_points3(axes[5],targets[lower],union[lower],observers[observers[:,2]<cut])
        titles[5]=f"F  Open cutaway below z = {cut:g}"
        titles[1]="B  Conservative voxel isosurface"
    for ax,title in zip(axes,titles):
        ax.set_title(title,loc="left",color=INK,pad=13,fontweight="medium")
    family=metadata["family"].replace("_"," ")
    fig.suptitle(f"{metadata['dimension']}D {metadata['geometry']} · {family} · seed {metadata['seed']}",
                 x=.045,y=.99,ha="left",fontsize=19,color=INK)
    fig.subplots_adjust(left=.04,right=.98,bottom=.10,top=.90,wspace=.20,hspace=.24)
    legend(fig)
    stem=output/folder.name
    save_figure(fig,stem,svg=metadata["dimension"]==2)
    return stem.with_suffix(".pdf")


def compact_pair(folders,output):
    """Two additional worlds, each shown as geometry / samples / visibility."""
    dimension=load_world(folders[0])[1]["dimension"]
    fig=plt.figure(figsize=(12,8.1))
    for row,folder in enumerate(folders):
        folder=Path(folder)
        world,m=load_world(folder)
        points=np.load(folder/"targets.npy");observers=np.load(folder/"observers.npy")
        matrix=load_npz(folder/"visibility_continuous.npz")
        visible=np.asarray(matrix.sum(axis=0)).ravel()>0
        for column in range(3):
            ax=fig.add_subplot(2,3,row*3+column+1,projection="3d" if dimension==3 else None)
            if dimension==2:
                axes2(ax,world,m["world_size"])
                if column:
                    if column==2:
                        ax.scatter(*points[~visible].T,s=2.4,c=MUTED,alpha=.35,rasterized=True)
                        ax.scatter(*points[visible].T,s=3.,c=TEAL,alpha=.65,rasterized=True)
                        ax.scatter(*observers.T,s=45,c=ORANGE,edgecolors="white",linewidths=.8)
                    else:
                        ax.scatter(*points.T,s=2.3,c=TEAL,alpha=.4,rasterized=True)
            else:
                axes3(ax,m["world_size"])
                vertices,faces=mesh_world(folder,world,m)
                if column==0:
                    draw_mesh(ax,vertices,faces,alpha=.24);obstacle_meshes(ax,world)
                elif column==1:
                    draw_points3(ax,points)
                else:
                    draw_mesh(ax,vertices,faces,alpha=.035);draw_points3(ax,points,visible,observers)
            label=["Continuous geometry","Volume samples" if dimension==3 else "Interior samples",
                   f"Eight observers · {visible.mean():.1%}"][column]
            ax.set_title(label,loc="left",pad=12,color=INK)
        fig.text(.045,.96-row*.43,
                 f"{m['family'].replace('_',' ').title()} · seed {m['seed']} · "
                 f"{m['room_count_achieved']} rooms · {m['obstacle_count_achieved']} obstacles",
                 fontsize=14,color=INK)
    fig.subplots_adjust(left=.035,right=.98,top=.91,bottom=.09,hspace=.29,wspace=.18)
    legend(fig);save_figure(fig,output,svg=dimension==2)
