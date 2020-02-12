use rand::{rngs::StdRng, SeedableRng};
use sha2::{Digest, Sha256};

use sss::lattice::{
    keygen, Anke, AnkeIdentity, AnkePublic, AnkeSessionKeyPart, AnkeSessionKeyPartR, Boris,
    BorisIdentity, BorisPublic, BorisSessionKeyPart, BorisSessionKeyPartR, Init, SessionKeyPart,
    SessionKeyPartMix,
};

#[test]
fn test() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(key_exchange)
        .unwrap()
        .join()
        .unwrap()
}

fn key_exchange() {
    let session_key_sampler = SessionKeyPart::parallel_sampler::<StdRng>(2, 1024);
    let session_key_mix_sampler_anke =
        SessionKeyPartMix::<Anke>::parallel_sampler::<StdRng>(2, 1024);
    let session_key_mix_sampler_boris =
        SessionKeyPartMix::<Boris>::parallel_sampler::<StdRng>(2, 1024);
    let mut rng = StdRng::from_entropy();
    let init = Init::new(&mut rng);
    let (anke_pri, anke_pub) = keygen(&mut rng, &init);
    let (boris_pri, boris_pub) = keygen(&mut rng, &init);
    eprintln!("keygen done");
    let (anke_key_part, anke_random) = SessionKeyPart::generate(&session_key_sampler, &init);
    let (boris_key_part, boris_random) = SessionKeyPart::generate(&session_key_sampler, &init);
    let seeder = |v: &[u8]| {
        let digest = Sha256::digest(v);
        let mut seed = [0u8; 32];
        seed[..].copy_from_slice(&digest);
        seed
    };
    eprintln!("generate anke key part");
    let (anke_part_mix, _anke_c, _anke_d) =
        SessionKeyPartMix::<Anke>::generate::<rand_chacha::ChaChaRng, _, _>(
            seeder,
            &session_key_mix_sampler_anke,
            AnkePublic(&[], &anke_pub),
            BorisPublic(&[], &boris_pub),
            AnkeSessionKeyPart(&anke_key_part),
            BorisSessionKeyPart(&boris_key_part),
            AnkeIdentity(&anke_pri),
            AnkeSessionKeyPartR(&anke_random),
        );
    eprintln!("generate boris key part");
    let (boris_part_mix, _boris_c, _boris_d) =
        SessionKeyPartMix::<Boris>::generate::<rand_chacha::ChaChaRng, _, _>(
            seeder,
            &session_key_mix_sampler_boris,
            AnkePublic(&[], &anke_pub),
            BorisPublic(&[], &boris_pub),
            AnkeSessionKeyPart(&anke_key_part),
            BorisSessionKeyPart(&boris_key_part),
            BorisIdentity(&boris_pri),
            BorisSessionKeyPartR(&boris_random),
        );

    let reconsiliator = boris_part_mix.reconciliator();
    let anke_key = boris_part_mix.reconciliate(&reconsiliator);
    let boris_key = anke_part_mix.reconciliate(&reconsiliator);
    assert_eq!(anke_key, boris_key);
    eprintln!("shared key: {:?}", anke_key);
}
