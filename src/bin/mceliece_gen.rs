use std::{fs::File, io::Write, path::PathBuf};

use rand::rngs::OsRng;
use sss::{
    artin::GF65536NPreparedMultipointEvalVZG,
    galois::{GF65536NTower, GF65536N},
    goppa::{generate, BinaryPacked},
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short = "e")]
    pubkey: PathBuf,
    #[structopt(short = "d")]
    prikey: PathBuf,
}

#[derive(thiserror::Error, Debug)]
enum Error {
    #[error("serialization: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

fn main() -> Result<(), Error> {
    let opt = Opt::from_args();
    let n = 6960;
    let t = 94;
    let (enc, dec) = generate::<GF65536N, GF65536NTower, GF65536NPreparedMultipointEvalVZG, _>(
        &mut OsRng, t, n, 32, 64,
    );
    write!(
        File::create(opt.pubkey)?,
        "{}",
        serde_json::to_string(&BinaryPacked(enc))?
    )?;
    write!(
        File::create(opt.prikey)?,
        "{}",
        serde_json::to_string(&BinaryPacked(dec))?
    )?;
    Ok(())
}
