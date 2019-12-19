use std::{borrow::Borrow, hash::Hash, marker::PhantomData};

#[derive(Clone, Debug, Hash, PartialEq, PartialOrd, Ord, Eq)]
pub enum MerkleTree<T, S, H, Ho> {
    Node {
        left: Box<Self>,
        right: Box<Self>,
        depth: usize,
        hash: Ho,
    },
    Leaf {
        data: Option<T>,
        proof: Option<S>,
        hash: Option<Ho>,
        _p: PhantomData<fn() -> H>,
    },
}
impl<T, S, H, Ho> MerkleTree<T, S, H, Ho> {
    fn make_leaf_hash(&mut self, hasher: &H)
    where
        H: MerkleHash<T, Output = Ho>
            + MerkleHash<S, Output = Ho>
            + MerkleHash<(Ho, Ho), Output = Ho>,
        Ho: Eq + Hash + Clone,
    {
        match self {
            Self::Leaf {
                data, proof, hash, ..
            } => {
                let data = <H as MerkleHash<T>>::hash(
                    hasher,
                    data.as_ref().expect("data should be present"),
                );
                let proof = <H as MerkleHash<S>>::hash(
                    hasher,
                    proof.as_ref().expect("proof should be present"),
                );
                *hash = Some(hasher.hash((data, proof)));
            }
            _ => panic!("logical error: expecting a leaf"),
        }
    }
    fn new_node(mut left: Self, mut right: Self, hasher: &H) -> Self
    where
        H: MerkleHash<T, Output = Ho>
            + MerkleHash<S, Output = Ho>
            + MerkleHash<(Ho, Ho), Output = Ho>,
        Ho: Eq + Hash + Clone,
    {
        let depth = match (&left, &right) {
            (
                Self::Node {
                    depth: depth_left, ..
                },
                Self::Node {
                    depth: depth_right, ..
                },
            ) => {
                assert_eq!(depth_left, depth_right);
                depth_left + 1
            }
            (Self::Leaf { .. }, Self::Leaf { .. }) => 1,
            _ => panic!("logical error: depth mismatch"),
        };
        if let Self::Leaf { hash: None, .. } = left {
            left.make_leaf_hash(hasher)
        }
        if let Self::Leaf { hash: None, .. } = right {
            right.make_leaf_hash(hasher)
        }
        let hash = match (&mut left, &mut right) {
            (
                Self::Leaf {
                    hash: hash_left, ..
                },
                Self::Leaf {
                    hash: hash_right, ..
                },
            ) => {
                let hash_left = hash_left.as_ref().unwrap().clone();
                let hash_right = hash_right.as_ref().unwrap().clone();
                hasher.hash((hash_left, hash_right))
            }
            (
                Self::Node {
                    hash: hash_left, ..
                },
                Self::Node {
                    hash: hash_right, ..
                },
            ) => {
                let hash_left = hash_left.clone();
                let hash_right = hash_right.clone();
                hasher.hash((hash_left, hash_right))
            }
            _ => panic!("unexpected left right tree nodes"),
        };
        let left = Box::new(left);
        let right = Box::new(right);
        Self::Node {
            left,
            right,
            depth,
            hash,
        }
    }

    pub fn new_leaf(data: T, proof: S) -> Self {
        Self::Leaf {
            data: Some(data),
            proof: Some(proof),
            hash: None,
            _p: PhantomData,
        }
    }

    pub fn build_tree(leaves: Vec<Self>, hasher: &H) -> Self
    where
        H: MerkleHash<T, Output = Ho>
            + MerkleHash<S, Output = Ho>
            + MerkleHash<(Ho, Ho), Output = Ho>,
        Ho: Eq + Hash + Clone,
    {
        let mut len = leaves.len();
        assert!(len > 1);
        assert_eq!((len - 1) & len, 0);
        assert!(leaves.iter().all(|node| match node {
            Self::Leaf { .. } => true,
            _ => false,
        }));
        let mut nodes = leaves;
        while nodes.len() > 1 {
            let mut nodes_itr = nodes.into_iter();
            let mut new_nodes = vec![];
            len >>= 1;
            for _ in 0..len {
                let left = nodes_itr
                    .next()
                    .expect("number of leaves should be power of 2");
                let right = nodes_itr
                    .next()
                    .expect("number of leaves should be power of 2");
                new_nodes.push(Self::new_node(left, right, hasher));
            }
            nodes = new_nodes;
        }
        nodes.pop().unwrap()
    }
}

pub trait MerkleHash<T> {
    type Output: Eq + Hash;
    fn hash<S: Borrow<T>>(&self, data: S) -> Self::Output;
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum MerkleTreeNodeType {
    Left,
    Right,
}

pub struct MerkleTreePath<'a, 'b, T, S, H, Ho>
where
    H: MerkleHash<T, Output = Ho> + MerkleHash<S, Output = Ho>,
    Ho: Eq + Hash,
{
    data: &'a T,
    proof: &'b S,
    path: Vec<(Ho, MerkleTreeNodeType)>,
    _p: PhantomData<fn() -> H>,
}

impl<'a, 'b, T, S, H, Ho> MerkleTreePath<'a, 'b, T, S, H, Ho>
where
    H: MerkleHash<T, Output = Ho> + MerkleHash<S, Output = Ho> + MerkleHash<(Ho, Ho), Output = Ho>,
    Ho: Eq + Hash,
{
    pub fn new(data: &'a T, proof: &'b S, path: Vec<(Ho, MerkleTreeNodeType)>) -> Self {
        Self {
            data,
            proof,
            path,
            _p: PhantomData,
        }
    }

    pub fn root_hash(&self, hasher: &H) -> Ho
    where
        Ho: Clone,
    {
        let tuple: (Ho, Ho) = (
            <H as MerkleHash<T>>::hash(&hasher, self.data),
            <H as MerkleHash<S>>::hash(&hasher, self.proof),
        );
        let mut hash = hasher.hash(tuple);
        for (sibling_hash, sibling_type) in &self.path {
            let tuple = match sibling_type {
                MerkleTreeNodeType::Left => (hash, sibling_hash.clone()),
                MerkleTreeNodeType::Right => (sibling_hash.clone(), hash),
            };
            hash = hasher.hash(tuple);
        }
        hash
    }
}

pub enum MerkleTreeQuorum<T, S, H, Ho>
where
    H: MerkleHash<T, Output = Ho> + MerkleHash<S, Output = Ho>,
    Ho: Eq + Hash,
{
    Leaf {
        hash: Ho,
    },
    Node {
        hash: Ho,
        left: Box<Self>,
        right: Box<Self>,
        depth: usize,
    },
    Undetermined {
        state: Option<(usize, Ho)>,
        _p: PhantomData<fn() -> (T, S, H)>,
    },
}

pub enum Error {
    Mismatch,
}

impl<T, S, H, Ho> MerkleTreeQuorum<T, S, H, Ho>
where
    H: MerkleHash<T, Output = Ho> + MerkleHash<S, Output = Ho> + MerkleHash<(Ho, Ho), Output = Ho>,
    Ho: Eq + Hash + Clone,
    T: Eq,
    S: Eq,
{
    pub fn new() -> Self {
        Self::Undetermined {
            state: None,
            _p: PhantomData,
        }
    }

    fn leaf(data: &T, proof: &S, hasher: &H) -> Self {
        let tuple: (Ho, Ho) = (
            <H as MerkleHash<T>>::hash(&hasher, data),
            <H as MerkleHash<S>>::hash(&hasher, proof),
        );
        let hash = hasher.hash(tuple);
        Self::Leaf { hash }
    }

    fn try_admit_rec(
        &mut self,
        hasher: &H,
        path: MerkleTreePath<T, S, H, Ho>,
    ) -> Result<(), Error> {
        let MerkleTreePath {
            data,
            proof,
            mut path,
            ..
        } = path;
        match self {
            Self::Undetermined {
                state: Some((depth, hash)),
                ..
            } if *depth == path.len() => {
                *self = match path.pop() {
                    None => Self::leaf(data, proof, hasher),
                    Some((hash_sibling, sibling_type)) => {
                        let depth = path.len() + 1;
                        let mut node = Self::new();
                        node.try_admit_rec(
                            hasher,
                            MerkleTreePath {
                                data,
                                proof,
                                path,
                                _p: PhantomData,
                            },
                        )?;
                        let hash_child = match &node {
                            Self::Node {
                                hash: hash_child, ..
                            } => hash_child.clone(),
                            Self::Leaf {
                                hash: hash_child, ..
                            } => hash_child.clone(),
                            _ => return Err(Error::Mismatch),
                        };

                        let hash_expected = {
                            let tuple = match sibling_type {
                                MerkleTreeNodeType::Left => {
                                    (hash_child.clone(), hash_sibling.clone())
                                }
                                MerkleTreeNodeType::Right => {
                                    (hash_sibling.clone(), hash_child.clone())
                                }
                            };
                            hasher.hash(tuple)
                        };
                        if *hash != hash_expected {
                            return Err(Error::Mismatch);
                        }
                        let sibling = Box::new(Self::Undetermined {
                            state: Some((depth - 1, hash_sibling)),
                            _p: PhantomData,
                        });
                        match sibling_type {
                            MerkleTreeNodeType::Left => Self::Node {
                                hash: hash.clone(),
                                depth,
                                left: Box::new(node),
                                right: sibling,
                            },
                            MerkleTreeNodeType::Right => Self::Node {
                                hash: hash.clone(),
                                depth,
                                right: Box::new(node),
                                left: sibling,
                            },
                        }
                    }
                };
                Ok(())
            }
            Self::Undetermined { state: None, .. } => {
                *self = match path.pop() {
                    None => Self::leaf(data, proof, hasher),
                    Some((hash_sibling, node_type)) => {
                        let depth = path.len() + 1;
                        let mut node = Self::new();
                        node.try_admit_rec(
                            hasher,
                            MerkleTreePath {
                                data,
                                proof,
                                path,
                                _p: PhantomData,
                            },
                        )?;
                        let hash_child = match &node {
                            Self::Node {
                                hash: hash_child, ..
                            } => hash_child.clone(),
                            Self::Leaf {
                                hash: hash_child, ..
                            } => hash_child.clone(),
                            _ => return Err(Error::Mismatch),
                        };

                        let hash = {
                            let tuple = match node_type {
                                MerkleTreeNodeType::Left => {
                                    (hash_child.clone(), hash_sibling.clone())
                                }
                                MerkleTreeNodeType::Right => {
                                    (hash_sibling.clone(), hash_child.clone())
                                }
                            };
                            hasher.hash(tuple)
                        };
                        let sibling = Box::new(Self::Undetermined {
                            state: Some((depth - 1, hash_sibling)),
                            _p: PhantomData,
                        });
                        match node_type {
                            MerkleTreeNodeType::Left => Self::Node {
                                hash: hash.clone(),
                                depth,
                                left: Box::new(node),
                                right: sibling,
                            },
                            MerkleTreeNodeType::Right => Self::Node {
                                hash: hash.clone(),
                                depth,
                                right: Box::new(node),
                                left: sibling,
                            },
                        }
                    }
                };
                Ok(())
            }
            Self::Leaf { hash: hash_quorom } => {
                let tuple: (Ho, Ho) = (
                    <H as MerkleHash<T>>::hash(&hasher, data),
                    <H as MerkleHash<S>>::hash(&hasher, proof),
                );
                let hash = hasher.hash(tuple);
                if *hash_quorom == hash && path.is_empty() {
                    Ok(())
                } else {
                    Err(Error::Mismatch)
                }
            }
            Self::Node {
                left, right, depth, ..
            } => match path.pop() {
                None => Err(Error::Mismatch),
                Some((hash_sibling, node_type)) => {
                    let (child, sibling) = match node_type {
                        MerkleTreeNodeType::Left => (&mut *right, &mut *left),
                        MerkleTreeNodeType::Right => (&mut *left, &mut *right),
                    };
                    match sibling.as_ref() {
                        Self::Leaf {
                            hash: sibling_hash_expected,
                            ..
                        } if *depth == 1 && *sibling_hash_expected == hash_sibling => child
                            .try_admit_rec(
                                hasher,
                                MerkleTreePath {
                                    data,
                                    proof,
                                    path,
                                    _p: PhantomData,
                                },
                            ),
                        Self::Node {
                            depth: sibling_depth,
                            hash: sibling_hash_expected,
                            ..
                        } if sibling_depth + 1 == *depth
                            && *sibling_hash_expected == hash_sibling =>
                        {
                            child.try_admit_rec(
                                hasher,
                                MerkleTreePath {
                                    data,
                                    proof,
                                    path,
                                    _p: PhantomData,
                                },
                            )
                        }
                        Self::Undetermined {
                            state: Some((sibling_depth, sibling_hash_expected)),
                            ..
                        } if sibling_depth + 1 == *depth
                            && *sibling_hash_expected == hash_sibling =>
                        {
                            child.try_admit_rec(
                                hasher,
                                MerkleTreePath {
                                    data,
                                    proof,
                                    path,
                                    _p: PhantomData,
                                },
                            )
                        }
                        _ => Err(Error::Mismatch),
                    }
                }
            },
            _ => Err(Error::Mismatch),
        }
    }

    pub fn try_admit(
        &mut self,
        hasher: &H,
        path: MerkleTreePath<T, S, H, Ho>,
    ) -> Result<(), Error> {
        self.try_admit_rec(hasher, path)
    }
}
