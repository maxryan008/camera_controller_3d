use std::{env, thread, time, f64::consts::PI, fs};
use rand::Rng;
use std::path::Path;
use bevy::reflect::{GetPath, TypePath, TypeUuid, Uuid};
use std::any::{Any, type_name};
use std::cmp::max;
use std::f32::MAX;
use futures_lite::future;
use bevy::{
    core_pipeline::{
        contrast_adaptive_sharpening::ContrastAdaptiveSharpeningSettings,
        experimental::taa::{
            TemporalAntiAliasBundle, TemporalAntiAliasPlugin, TemporalAntiAliasSettings,
        },
        fxaa::{Fxaa, Sensitivity},
    },
    asset::LoadState,
    pbr::wireframe::{Wireframe, WireframeConfig, WireframePlugin},
    tasks::{AsyncComputeTaskPool, Task},
    prelude::*,
    render::{render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin},
};
use bevy::app::AppLabel;
use bevy::ecs::system::IntoSystem;
use bevy::input::mouse::MouseMotion;
use bevy::math::{vec2, vec3};
use bevy::prelude::KeyCode::C;
use bevy::window::{WindowMode,CursorGrabMode,PrimaryWindow};
use noise::{Abs, NoiseFn, Perlin, Seedable};
use bevy::render::mesh::{self, PrimitiveTopology};
use bevy::render::render_resource::TextureUsages;
use bevy::ui::FocusPolicy::Block;
use bevy::utils::hashbrown::HashMap;
use bevy_asset::AssetPathId;
use noise::core::perlin::perlin_3d;

const RENDER_DISTANCE: f32 = 2.;
const CAM_MOVE_SPEED: f32 = 10.;
const PERLIN_OCTAVE_2_MOD: f64 = 2.;
const PERLIN_FULL_MOD: f64 = 0.001;
const CHUNK_SIZE: i32 = 16;
const MANTLE: f64 = 0.8;
const FACES: [[f64; 3]; 6] = [[0.0,0.0,-1.0],[0.0,0.0,1.0],[0.0,1.0,0.0],[0.0,-1.0,0.0],[-1.0,0.0,0.0],[1.0,0.0,0.0]];
const VOXEL_VERTS: [[f32; 3]; 8] = [[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0]];
const VOXEL_TRIS: [[usize; 6]; 6] = [[0,3,1,1,3,2],[5,6,4,4,6,7],[3,7,2,2,7,6],[1,5,0,0,5,4],[4,7,0,0,7,3],[1,2,5,5,2,6]];
const WORLD_SIZE: [i32; 3] = [3584,1024,3584];

#[derive(Component)]
struct MyGameCamera {
    move_speed: f32,
}

#[derive(Component, Debug)]
struct ChunkComponent {
    ratio: i32,
    pos: (i32,i32,i32),
}

#[derive(Clone)]
struct ChunkData {
    voxel_array: Vec<Voxel>,
    pos: (i32,i32,i32),
    ratio: i32,
}

#[derive(Clone)]
struct Voxel {
    voxel_type: i32,
    solid: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, States)]
enum AppState {
    #[default]
    Setup,
    Finished,
    Generating,
}

#[derive(Resource, Default)]
struct TextureHandles {
    handles: Vec<HandleUntyped>,
}

#[derive(Resource, Default)]
struct TextureAtlasBuilt {
    tex: Handle<Image>,
    tex_rects: Vec<Rect>,
    tex_map: Vec<usize>,
    tex_size: Vec2,

}

#[derive(Resource)]
struct Biomes {
    biome_pointer_array: [usize; 121]
}

#[derive(Resource, Default)]
struct EntityHashMap(HashMap<i32,Entity>);

struct RegionalData {
    region_chunks: Vec<ChunkData>,
}

struct RenderData {
    data : (Vec<[f32; 3]>,Vec<u32>,Vec<[f32; 2]>),
    ratio : i32,
    pos: (i32,i32,i32),
}

#[derive(Resource, Default)]
struct CameraTransform {
    transform : Transform,
}

#[derive(Component)]
struct ComputeRegion(Task<RegionalData>);

#[derive(Component)]
struct ComputeRender(Task<RenderData>);

#[derive(Component)]
struct ComputeChunk(Task<ChunkData>);

fn setup(
    mut next_state: ResMut<NextState<AppState>>,
    asset_server: Res<AssetServer>,
    texture_handles: Res<TextureHandles>,
    mut texture_atlases: ResMut<Assets<TextureAtlas>>,
    mut textures: ResMut<Assets<Image>>,
    mut commands: Commands,
    mut wireframe_config: ResMut<WireframeConfig>,
    mut texture_atlas_data : ResMut<TextureAtlasBuilt>,
)
{
    //enable wireframe?
    wireframe_config.global = false;

    //build texture atlas
    let mut texture_atlas_builder = TextureAtlasBuilder::default();
    for handle in &texture_handles.handles {
        let handle = handle.typed_weak();
        let Some(texture) = textures.get(&handle) else {
            warn!("{:?} did not resolve to an `Image` asset.", asset_server.get_handle_path(handle));
            continue;
        };
        texture_atlas_builder.add_texture(handle, texture);
    }

    //assign texture atlas
    let texture_atlas = texture_atlas_builder.finish(&mut textures).unwrap();
    texture_atlas_data.tex = texture_atlas.texture.clone();

    //texture
    for texhand in texture_atlas.texture_handles.iter(){
        //load texs into map
        let mut tex_map: Vec<usize> = Vec::new();
        tex_map.push(0);
        for line in fs::read_to_string("Assets/TexMem").unwrap().lines() {
            for (key,val) in texhand {
                let path = asset_server.get_handle_path(key).unwrap();
                if path.path().file_name().unwrap().to_string_lossy().split(".png").next().unwrap() == line {
                    tex_map.push(*val);
                }
            }

        }
        texture_atlas_data.tex_rects = texture_atlas.textures.clone();
        texture_atlas_data.tex_map = tex_map;
        texture_atlas_data.tex_size = texture_atlas.size;
    }

    //biomes pointer array init
    let mut biome_pointer_array: [usize; 121] = [0; 121];
    let binding = fs::read_to_string("Assets/BiomesMem").unwrap();
    for (pos, line) in binding.lines().enumerate() {
        if pos != 0 {
            let biome_dat: Vec<&str> = line.split(":").collect();
            for num in biome_dat[1].split(",").collect::<Vec<&str>>() {
                let pointer = num.to_string().parse::<usize>().unwrap();
                biome_pointer_array[pointer] = pos;
            }
        }
    }
    commands.insert_resource(Biomes{biome_pointer_array});

    commands.spawn((
        Camera3dBundle{
            transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        MyGameCamera{
            move_speed: CAM_MOVE_SPEED,
        },
    ));
    next_state.set(AppState::Generating);
}

fn fly_cam(
    keys: Res<Input<KeyCode>>,
    mut windows: Query<&mut Window>,
    buttons: Res<Input<MouseButton>>,
    mut motion_evr: EventReader<MouseMotion>,
    mut query: Query<(&mut MyGameCamera, &mut Transform)>,
    timer: Res<Time>,
)
{
    let mut window = windows.single_mut();
    let mut rot = Vec2::ZERO;
    if keys.pressed(KeyCode::W) {
        for (camera, mut transform) in &mut query {
            let forward = transform.forward();
            transform.translation += forward * camera.move_speed * timer.delta_seconds();
        }
    }
    if keys.pressed(KeyCode::S) {
        for (camera, mut transform) in &mut query {
            let back = transform.back();
            transform.translation += back * camera.move_speed * timer.delta_seconds();
        }
    }
    if keys.pressed(KeyCode::A) {
        for (camera, mut transform) in &mut query {
            let left = transform.left();
            transform.translation += left * camera.move_speed * timer.delta_seconds();
        }
    }
    if keys.pressed(KeyCode::D) {
        for (camera, mut transform) in &mut query {
            let right = transform.right();
            transform.translation += right * camera.move_speed * timer.delta_seconds();
        }
    }
    if keys.pressed(KeyCode::Space) {
        for (camera, mut transform) in &mut query {
            transform.translation.y += camera.move_speed * timer.delta_seconds();
        }
    }
    if keys.pressed(KeyCode::ShiftLeft) {
        for (camera, mut transform) in &mut query {
            transform.translation.y -= camera.move_speed * timer.delta_seconds();
        }
    }
    for ev in motion_evr.iter() {
        rot += ev.delta;
    }

    for (camera, mut transform) in &mut query {
        let delta_x = rot.x / window.width() * std::f32::consts::PI * 2.0;
        let delta_y = rot.y / window.height() * std::f32::consts::PI;
        let yaw = Quat::from_rotation_y(-delta_x);
        let pitch = Quat::from_rotation_x(-delta_y);
        transform.rotation = yaw * transform.rotation; // rotate around global y axis
        transform.rotation = transform.rotation * pitch; // rotate around local x axis
    }


    if buttons.just_pressed(MouseButton::Left) {
        window.cursor.visible = false;
        window.cursor.grab_mode = CursorGrabMode::Locked;
    }

    if keys.just_pressed(KeyCode::Escape) {
        window.cursor.visible = true;
        window.cursor.grab_mode = CursorGrabMode::None;
    }
}

fn block_to_tex(
    block_type: i32,
    tex_map: Vec<usize>,
    tex_rects: Vec<Rect>,
    tex_size: Vec2,
) -> Rect
{

    return Rect::new(tex_rects[tex_map[block_type as usize]].min.x/tex_size.x,tex_rects[tex_map[block_type as usize]].min.y/tex_size.y,tex_rects[tex_map[block_type as usize]].max.x/tex_size.x,tex_rects[tex_map[block_type as usize]].max.y/tex_size.y);
}

fn calculate_chunk_data (
    pos: (i32,i32,i32),
    region_ratio: i32,
    biome_pointer_array: [usize; 121],
) -> Vec<Voxel>
{
    let mut voxel_array: Vec<Voxel> = Vec::new();
    let biome_file = fs::read_to_string("Assets/BiomesMem").unwrap();
    let biome_lines = biome_file.lines().collect::<Vec<&str>>();
    for x in 0 .. 16 {
        for y in 0 .. 16 {
            for z in 0 .. 16 {
                let mut perlin = Perlin::new(932843);
                let perlin_temp = perlin.get([(x*region_ratio + pos.0*16) as f64 * PERLIN_FULL_MOD,(z*region_ratio + pos.2*16) as f64 * PERLIN_FULL_MOD]) as f32;
                perlin = Perlin::new(1283024734);
                let perlin_hum = perlin.get([(x*region_ratio + pos.0*16) as f64 * PERLIN_FULL_MOD * PERLIN_OCTAVE_2_MOD,(z*region_ratio + pos.2*16) as f64 * PERLIN_FULL_MOD * PERLIN_OCTAVE_2_MOD]) as f32;
                let line = biome_pointer_array[(((perlin_temp+1.)*55.+((perlin_hum+1.)*5.5)).round() - 1.) as usize];
                let biome = biome_lines[line];
                let biome_dat : Vec<&str> = biome.split(":").collect();
                voxel_array.push(Voxel {
                    voxel_type: biome_dat[3].parse().unwrap(),
                    solid: true,
                });
            }
        }
    }
    return voxel_array;
}

fn recalculate_chunk (
    position: (i32,i32,i32),
    region_ratio: i32,
    biome_pointer_array: [usize; 121],
) -> Vec<ChunkData>
{
    let mut chunks: Vec<ChunkData> = Vec::new();
    let voxel_array: Vec<Voxel> = calculate_chunk_data((position.0/16, position.1/16, position.2/16), region_ratio, biome_pointer_array);
    chunks.push(ChunkData {
        voxel_array,
        pos: (position.0, position.1, position.2),
        ratio: region_ratio,
    });
    return chunks;
}

fn calculate_region_data (
    outside_region: Vec2,
    inside_region: Vec2,
    center: Vec3,
    region_ratio: i32,
    biome_pointer_array: [usize; 121],
) -> Vec<ChunkData>
{
    let camera_x = center.x as i32;
    let camera_y = center.y as i32;
    let camera_z = center.z as i32;
    let outside_region_max = outside_region.x as i32;
    let outside_region_min = outside_region.y as i32;
    let inside_region_max = inside_region.x as i32;
    let inside_region_min = inside_region.y as i32;
    let mut chunks: Vec<ChunkData> = Vec::new();

    /*
    for x in (outside_region_min .. outside_region_max).step_by(region_ratio as usize) {
        for y in (inside_region_max .. outside_region_max).step_by(region_ratio as usize) {
            for z in (outside_region_min .. outside_region_max).step_by(region_ratio as usize) {
                //top
                if x <= WORLD_SIZE[0]/2 && x >= WORLD_SIZE[0]/-2 && y <= WORLD_SIZE[1]/2 && y >= WORLD_SIZE[1]/-2 && z <= WORLD_SIZE[2]/2 && z >= WORLD_SIZE[2]/-2 {
                    let voxel_array: Vec<Voxel> = calculate_chunk_data((x+camera_x/16, y+camera_y/16, z+camera_z/16), region_ratio, biome_pointer_array);
                    chunks.push(ChunkData {
                        voxel_array,
                        pos: (x * CHUNK_SIZE, y * CHUNK_SIZE, z * CHUNK_SIZE),
                        ratio: region_ratio,
                    })
                }
            }
        }
    }
    */
    for x in (outside_region_min .. outside_region_max).step_by(region_ratio as usize) {
        for y in (inside_region_min .. inside_region_max).step_by(region_ratio as usize) {
            for z in (inside_region_max .. outside_region_max).step_by(region_ratio as usize) {
                //back
                if x <= WORLD_SIZE[0]/2 && x >= WORLD_SIZE[0]/-2 && y <= WORLD_SIZE[1]/2 && y >= WORLD_SIZE[1]/-2 && z <= WORLD_SIZE[2]/2 && z >= WORLD_SIZE[2]/-2 {
                    let voxel_array: Vec<Voxel> = calculate_chunk_data((x+camera_x/16, y+camera_y/16, z+camera_z/16), region_ratio, biome_pointer_array);
                    chunks.push(ChunkData {
                        voxel_array,
                        pos: (x * CHUNK_SIZE, y * CHUNK_SIZE, z * CHUNK_SIZE),
                        ratio: region_ratio,
                    })
                }
            }
        }
    }
    for x in (outside_region_min .. outside_region_max).step_by(region_ratio as usize) {
        for y in (outside_region_min .. inside_region_min).step_by(region_ratio as usize) {
            for z in (outside_region_min .. outside_region_max).step_by(region_ratio as usize) {
                //bottom
                if x <= WORLD_SIZE[0]/2 && x >= WORLD_SIZE[0]/-2 && y <= WORLD_SIZE[1]/2 && y >= WORLD_SIZE[1]/-2 && z <= WORLD_SIZE[2]/2 && z >= WORLD_SIZE[2]/-2 {
                    let voxel_array: Vec<Voxel> = calculate_chunk_data((x+camera_x/16, y+camera_y/16, z+camera_z/16), region_ratio, biome_pointer_array);
                    chunks.push(ChunkData {
                        voxel_array,
                        pos: (x * CHUNK_SIZE, y * CHUNK_SIZE, z * CHUNK_SIZE),
                        ratio: region_ratio,
                    })
                }
            }
        }
    }
    for x in (outside_region_min .. outside_region_max).step_by(region_ratio as usize) {
        for y in (inside_region_min .. inside_region_max).step_by(region_ratio as usize) {
            for z in (outside_region_min .. inside_region_min).step_by(region_ratio as usize) {
                //front
                if x <= WORLD_SIZE[0]/2 && x >= WORLD_SIZE[0]/-2 && y <= WORLD_SIZE[1]/2 && y >= WORLD_SIZE[1]/-2 && z <= WORLD_SIZE[2]/2 && z >= WORLD_SIZE[2]/-2 {
                    let voxel_array: Vec<Voxel> = calculate_chunk_data((x+camera_x/16, y+camera_y/16, z+camera_z/16), region_ratio, biome_pointer_array);
                    chunks.push(ChunkData {
                        voxel_array,
                        pos: (x * CHUNK_SIZE, y * CHUNK_SIZE, z * CHUNK_SIZE),
                        ratio: region_ratio,
                    })
                }
            }
        }
    }
    for x in (inside_region_max .. outside_region_max).step_by(region_ratio as usize) {
        for y in (inside_region_min .. inside_region_max).step_by(region_ratio as usize) {
            for z in (inside_region_min .. inside_region_max).step_by(region_ratio as usize) {
                //right
                if x <= WORLD_SIZE[0]/2 && x >= WORLD_SIZE[0]/-2 && y <= WORLD_SIZE[1]/2 && y >= WORLD_SIZE[1]/-2 && z <= WORLD_SIZE[2]/2 && z >= WORLD_SIZE[2]/-2 {
                    let voxel_array: Vec<Voxel> = calculate_chunk_data((x+camera_x/16, y+camera_y/16, z+camera_z/16), region_ratio, biome_pointer_array);
                    chunks.push(ChunkData {
                        voxel_array,
                        pos: (x * CHUNK_SIZE, y * CHUNK_SIZE, z * CHUNK_SIZE),
                        ratio: region_ratio,
                    })
                }
            }
        }
    }
    for x in (outside_region_min .. inside_region_min).step_by(region_ratio as usize) {
        for y in (inside_region_min .. inside_region_max).step_by(region_ratio as usize) {
            for z in (inside_region_min .. inside_region_max).step_by(region_ratio as usize) {
                //left
                if x <= WORLD_SIZE[0]/2 && x >= WORLD_SIZE[0]/-2 && y <= WORLD_SIZE[1]/2 && y >= WORLD_SIZE[1]/-2 && z <= WORLD_SIZE[2]/2 && z >= WORLD_SIZE[2]/-2 {
                    let voxel_array: Vec<Voxel> = calculate_chunk_data((x+camera_x/16, y+camera_y/16, z+camera_z/16), region_ratio, biome_pointer_array);
                    chunks.push(ChunkData {
                        voxel_array,
                        pos: (x * CHUNK_SIZE, y * CHUNK_SIZE, z * CHUNK_SIZE),
                        ratio: region_ratio,
                    })
                }
            }
        }
    }
    return chunks;
}

fn task_generation(
    mut commands: Commands,
    biome_data: Res<Biomes>,
    query: Query<(&MyGameCamera, &Transform)>,
)
{
    let thread_pool = AsyncComputeTaskPool::get();
    println!("Generating Regions");
    let biome_pointer_array = biome_data.biome_pointer_array.clone();
    for (&ref camera, &transform) in &query {
        //region generator
        //generate region 1:1 render distance
        let regional_task = thread_pool.spawn(async move {
            RegionalData{region_chunks:calculate_region_data(vec2(RENDER_DISTANCE,RENDER_DISTANCE*-1.),vec2(0.,0.),transform.translation, 1, biome_pointer_array.clone())}
        });
        commands.spawn(ComputeRegion(regional_task));
        //generate region 1:2 2 x chunks
        let regional_task = thread_pool.spawn(async move {
            RegionalData{region_chunks:calculate_region_data(vec2(RENDER_DISTANCE*4.,RENDER_DISTANCE*-4.),vec2(RENDER_DISTANCE,RENDER_DISTANCE*-1.),transform.translation, 2, biome_pointer_array.clone())}
        });
        commands.spawn(ComputeRegion(regional_task));
        //generate region 1:4 4 x chunks
        let regional_task = thread_pool.spawn(async move {
            RegionalData{region_chunks:calculate_region_data(vec2(RENDER_DISTANCE*8.,RENDER_DISTANCE*-8.),vec2(RENDER_DISTANCE*4.,RENDER_DISTANCE*-4.),transform.translation, 4, biome_pointer_array.clone())}
        });
        commands.spawn(ComputeRegion(regional_task));
        //generate region 1:8 8 x chunks
        let regional_task = thread_pool.spawn(async move {
            RegionalData{region_chunks:calculate_region_data(vec2(RENDER_DISTANCE*16.,RENDER_DISTANCE*-16.),vec2(RENDER_DISTANCE*8.,RENDER_DISTANCE*-8.),transform.translation, 8, biome_pointer_array.clone())}
        });
        commands.spawn(ComputeRegion(regional_task));

        //generate region 1:16 16 x chunks
        let regional_task = thread_pool.spawn(async move {
            RegionalData{region_chunks:calculate_region_data(vec2(RENDER_DISTANCE*32.,RENDER_DISTANCE*-32.),vec2(RENDER_DISTANCE*16.,RENDER_DISTANCE*-16.),transform.translation, 16, biome_pointer_array.clone())}
        });
        commands.spawn(ComputeRegion(regional_task));
        //generate region 1:64 64 x chunks
        let regional_task = thread_pool.spawn(async move {
            RegionalData{region_chunks:calculate_region_data(vec2(RENDER_DISTANCE*128.,RENDER_DISTANCE*-128.),vec2(RENDER_DISTANCE*32.,RENDER_DISTANCE*-32.),transform.translation, 64, biome_pointer_array.clone())}
        });
        commands.spawn(ComputeRegion(regional_task));
        //generate region 1:256 256 x chunks
        let regional_task = thread_pool.spawn(async move {
            RegionalData{region_chunks:calculate_region_data(vec2((WORLD_SIZE[0] / 2) as f32, (WORLD_SIZE[0] / -2) as f32), vec2(RENDER_DISTANCE*128., RENDER_DISTANCE*-128.), transform.translation, 256, biome_pointer_array.clone())}
        });
        commands.spawn(ComputeRegion(regional_task));

    }
}

fn array_to_render (
    chunk: ChunkData,
    tex_map: Vec<usize>,
    tex_rects: Vec<Rect>,
    tex_size: Vec2,
) -> (Vec<[f32; 3]>,Vec<u32>,Vec<[f32; 2]>)
{
    let mut vertices : Vec<[f32; 3]> = Vec::new();
    let mut indices : Vec<u32> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let cubed_chunk_size= CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE;
    for (index, voxel) in chunk.voxel_array.iter().enumerate() {
        let z = (index as f64 % (CHUNK_SIZE) as f64) as f64;
        let y = (((index as f64 - z) / (CHUNK_SIZE) as f64) % (CHUNK_SIZE) as f64) as f64;
        let x = ((index as f64 - z - (CHUNK_SIZE) as f64 * y) / (CHUNK_SIZE * CHUNK_SIZE) as f64) as f64;
        //sides
        if voxel.voxel_type != 0 {
            for p in 0 .. 6 {
                let neighbour_index = (((x + FACES[p][0]) * CHUNK_SIZE as f64 * CHUNK_SIZE as f64) + ((y + FACES[p][1]) * CHUNK_SIZE as f64) + (z + FACES[p][2]));
                if (x + FACES[p][0]) >= 0. && (x + FACES[p][0]) < CHUNK_SIZE as f64 && (y + FACES[p][1]) >= 0. && (y + FACES[p][1]) < CHUNK_SIZE as f64 && (z + FACES[p][2]) >= 0. && (z + FACES[p][2]) < CHUNK_SIZE as f64 {
                    if chunk.voxel_array[neighbour_index as usize].solid == false {
                        for i in 0 .. 6 {
                            vertices.push([(VOXEL_VERTS[VOXEL_TRIS[p][i]][0] + x as f32) * chunk.ratio as f32 + chunk.pos.0 as f32,(VOXEL_VERTS[VOXEL_TRIS[p][i]][1] + y as f32) * chunk.ratio as f32 + chunk.pos.1 as f32,(VOXEL_VERTS[VOXEL_TRIS[p][i]][2] + z as f32) * chunk.ratio as f32 + chunk.pos.2 as f32]);
                            indices.push((vertices.len()-1) as u32);
                        }
                        let uv_rect = block_to_tex(voxel.voxel_type, tex_map.clone(), tex_rects.clone(),tex_size);
                        uvs.push([uv_rect.max.x,uv_rect.max.y]);
                        uvs.push([uv_rect.max.x,uv_rect.min.y]);
                        uvs.push([uv_rect.min.x,uv_rect.max.y]);
                        uvs.push([uv_rect.min.x,uv_rect.max.y]);
                        uvs.push([uv_rect.max.x,uv_rect.min.y]);
                        uvs.push([uv_rect.min.x,uv_rect.min.y]);
                    }
                }else{
                    for i in 0 .. 6 {
                        vertices.push([(VOXEL_VERTS[VOXEL_TRIS[p][i]][0] + x as f32) * chunk.ratio as f32 + chunk.pos.0 as f32,(VOXEL_VERTS[VOXEL_TRIS[p][i]][1] + y as f32) * chunk.ratio as f32 + chunk.pos.1 as f32,(VOXEL_VERTS[VOXEL_TRIS[p][i]][2] + z as f32) * chunk.ratio as f32 + chunk.pos.2 as f32]);
                        indices.push((vertices.len()-1) as u32);
                    }
                    let uv_rect = block_to_tex(voxel.voxel_type, tex_map.clone(), tex_rects.clone(),tex_size);
                    uvs.push([uv_rect.max.x,uv_rect.max.y]);
                    uvs.push([uv_rect.max.x,uv_rect.min.y]);
                    uvs.push([uv_rect.min.x,uv_rect.max.y]);
                    uvs.push([uv_rect.min.x,uv_rect.max.y]);
                    uvs.push([uv_rect.max.x,uv_rect.min.y]);
                    uvs.push([uv_rect.min.x,uv_rect.min.y]);
                }
            }
        }
    }



    return (vertices,indices,uvs);
}

fn handle_tasks(
    mut commands: Commands,
    mut region_tasks: Query<(Entity, &mut ComputeRegion)>,
    mut render_tasks: Query<(Entity, &mut ComputeRender)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    texture_atlas: Res<TextureAtlasBuilt>,
    texture_atlas_data: Res<TextureAtlasBuilt>,
)
{
    //handle data into render
    let thread_pool = AsyncComputeTaskPool::get();
    for (entity, mut task) in &mut region_tasks {
        if let Some(region_data) = future::block_on(future::poll_once(&mut task.0)) {
            for chunk in region_data.region_chunks {
                let tex_map = texture_atlas_data.tex_map.clone().to_vec();
                let tex_rects = texture_atlas_data.tex_rects.clone().to_vec();
                let tex_size = texture_atlas_data.tex_size.clone();
                let render_task = thread_pool.spawn(async move {
                    RenderData{
                        data : array_to_render(chunk.clone(),tex_map,tex_rects,tex_size),
                        ratio : chunk.ratio,
                        pos : chunk.pos,
                    }
                });
                commands.spawn(ComputeRender(render_task));
            }

            // Task is complete, so remove task component from entity
            commands.entity(entity).remove::<ComputeRegion>();
        }
    }

    //handle render into visual
    for (entity, mut task) in &mut render_tasks {
        if let Some(render_data) = future::block_on(future::poll_once(&mut task.0)) {

            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, render_data.data.0);
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0., 0., 1.]; render_data.data.1.len()]);
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, render_data.data.2);
            mesh.set_indices(Some(mesh::Indices::U32(render_data.data.1)));
            commands.spawn((
                    PbrBundle {
                mesh: meshes.add(mesh),
                material: materials.add(StandardMaterial{
                    emissive: Color::rgb(1.,1.,1.).into(),
                    emissive_texture: Option::from(texture_atlas.tex.clone()),
                    ..default()
                }),
                transform: Transform::from_xyz(0.,0.,0.),
                ..default()
            },
                    ChunkComponent {ratio: render_data.ratio, pos: render_data.pos},
            ));

            // Task is complete, so remove task component from entity
            commands.entity(entity).remove::<ComputeRender>();
        }
    }

}

fn load_textures(mut texture_handles: ResMut<TextureHandles>, asset_server: Res<AssetServer>) {
    texture_handles.handles = asset_server.load_folder("textures/blocks").unwrap();

}

fn check_textures(
    mut next_state: ResMut<NextState<AppState>>,
    texture_handles: Res<TextureHandles>,
    asset_server: Res<AssetServer>,
)
{
    if let LoadState::Loaded = asset_server
        .get_group_load_state(texture_handles.handles.iter().map(|handle| handle.id()))
    {
        next_state.set(AppState::Finished);
    }
}

fn task_updating(
    camera_query: Query<(&MyGameCamera, &Transform)>,
    chunk_query: Query<(Entity, &ChunkComponent)>,
    mut camera_transform: ResMut<CameraTransform>,
    mut commands: Commands,
    biome_data: Res<Biomes>,
)
{
    let thread_pool = AsyncComputeTaskPool::get();
    let biome_pointer_array = biome_data.biome_pointer_array.clone();
    for (&ref camera, &transform) in &camera_query  {
        if transform.translation != camera_transform.transform.translation {
            camera_transform.transform = transform;
            for (entity_id, chunk_component) in &chunk_query {
                let chunk_data = chunk_component.clone();
                let updated_x = (chunk_data.pos.0 as f32 - transform.translation.x).abs() as i32;
                let updated_y = (chunk_data.pos.1 as f32 - transform.translation.y).abs() as i32;
                let updated_z = (chunk_data.pos.2 as f32 - transform.translation.z).abs() as i32;
                let max_val = (max(max(updated_x,updated_y), updated_z)/16) as f32;
                let pos = chunk_data.pos.clone();
                if max_val >= RENDER_DISTANCE * 256. {
                    if chunk_data.ratio != 64 {
                        println!("64");
                        println!("{:?}",chunk_data);
                        //calculate new chunk
                        let regional_task = thread_pool.spawn(async move {
                            RegionalData{region_chunks:recalculate_chunk(pos,64, biome_pointer_array.clone())}
                        });
                        commands.spawn(ComputeRegion(regional_task));
                        //remove old chunk
                        commands.entity(entity_id).despawn();
                    }
                }else if max_val >= RENDER_DISTANCE * 64. {
                    if chunk_data.ratio != 16 {
                        println!("16");
                        println!("{:?}",chunk_data);
                        //calculate new chunk
                        let regional_task = thread_pool.spawn(async move {
                            RegionalData{region_chunks:recalculate_chunk(pos,16, biome_pointer_array.clone())}
                        });
                        commands.spawn(ComputeRegion(regional_task));
                        //remove old chunk
                        commands.entity(entity_id).despawn();
                    }
                }else if max_val >= RENDER_DISTANCE * 32. {
                    if chunk_data.ratio != 8 {
                        println!("8");
                        println!("{:?}",chunk_data);
                        //calculate new chunk
                        let regional_task = thread_pool.spawn(async move {
                            RegionalData{region_chunks:recalculate_chunk(pos,8, biome_pointer_array.clone())}
                        });
                        commands.spawn(ComputeRegion(regional_task));
                        //remove old chunk
                        commands.entity(entity_id).despawn();
                    }
                }else if max_val >= RENDER_DISTANCE * 16. {
                    if chunk_data.ratio != 4 {
                        println!("4");
                        println!("{:?}",chunk_data);
                        //calculate new chunk
                        let regional_task = thread_pool.spawn(async move {
                            RegionalData{region_chunks:recalculate_chunk(pos,4, biome_pointer_array.clone())}
                        });
                        commands.spawn(ComputeRegion(regional_task));
                        //remove old chunk
                        commands.entity(entity_id).despawn();
                    }
                }else if max_val >= RENDER_DISTANCE * 8. {
                    if chunk_data.ratio != 2 {
                        println!("2");
                        println!("{:?}",chunk_data);
                        //calculate new chunk
                        let regional_task = thread_pool.spawn(async move {
                            RegionalData{region_chunks:recalculate_chunk(pos,2, biome_pointer_array.clone())}
                        });
                        commands.spawn(ComputeRegion(regional_task));
                        //remove old chunk
                        commands.entity(entity_id).despawn();
                    }
                }else {
                    if chunk_data.ratio != 1 {
                        println!("1");
                        println!("{:?}",chunk_data);
                        //calculate new chunk
                        let regional_task = thread_pool.spawn(async move {
                            RegionalData{region_chunks:recalculate_chunk(pos,1, biome_pointer_array.clone())}
                        });
                        commands.spawn(ComputeRegion(regional_task));
                        //remove old chunk
                        commands.entity(entity_id).despawn();
                    }
                }
            }
        }
    }
}

fn main()
{
    App::new()
        .init_resource::<TextureHandles>()
        .init_resource::<TextureAtlasBuilt>()
        .init_resource::<CameraTransform>()
        .init_resource::<EntityHashMap>()
        .add_state::<AppState>()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin::default_nearest()).set(WindowPlugin {
                primary_window: Some(Window {
                    resizable: false,
                    mode: WindowMode::BorderlessFullscreen,
                    ..default()
                }),
                ..default()
            }),
            WireframePlugin,
            TemporalAntiAliasPlugin,
        ))
        .add_systems(OnEnter(AppState::Setup), load_textures)
        .add_systems(Update, check_textures.run_if(in_state(AppState::Setup)))
        .add_systems(OnEnter(AppState::Finished), setup)
        .add_systems(OnEnter(AppState::Generating), task_generation)
        .add_systems(Update, handle_tasks)
        .add_systems(Update, fly_cam)
        .add_systems(Update, task_updating)
        .run();

}