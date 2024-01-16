use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::thread;
use std::time::Duration;

#[derive(Serialize, Deserialize)]
struct ScreenshotRequest {
    screenshot: String,
    score: i32,
    game_ended: bool,
    player_position: f32,
    enemy_position: f32,
}

async fn upload_screenshot(request: web::Json<ScreenshotRequest>) -> impl Responder {
    let env_record = Path::new("env_record");
    fs::create_dir_all(env_record).unwrap();

    let record_path = env_record.join("record1");
    let mut file = File::create(&record_path).unwrap();
    file.write_all(serde_json::to_string(&*request).unwrap().as_bytes())
        .unwrap();

    let action_path = env_record.join("action1");
    while !action_path.exists() {
        thread::sleep(Duration::from_millis(10));
    }

    let action = fs::read_to_string(action_path).unwrap();
    HttpResponse::Ok().json({"action": action})
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        let cors = Cors::default()
            .allowed_origin("http://localhost:4000")
            .allowed_methods(vec!["GET", "POST"])
            .allowed_headers(vec!["*"])
            .supports_credentials();

            App::new()
                .wrap(cors)
                .route("/upload-screenshot", web::post().to(upload_screenshot))
        })
        .bind("localhost:8000")?
        .run()
        .await
    }                   
