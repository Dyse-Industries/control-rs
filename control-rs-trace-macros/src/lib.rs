//! `#[req]`: marks an item with the verification conditions it covers.
//!
//! The attribute checks that each argument is a qualified condition ID
//! without a tag and returns the item unchanged. It writes nothing: `trace-marks` finds markers
//! by reading source text, so the link does not depend on what a build
//! compiles.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{LitStr, Token};

/// The qualified IDs of one attribute, or the error to report.
type Ids = syn::Result<Vec<String>>;

/// Marks an item with the verification conditions it covers.
///
/// The arguments are qualified condition IDs, `<doc>#<id>`, as string
/// literals. The item is returned unchanged.
///
/// ```ignore
/// #[req("storage#VC-3.1", "storage#VC-4.1")]
/// #[test]
/// fn packed_value_bounds() {}
/// ```
///
/// An argument that is not a qualified ID fails to compile:
///
/// ```compile_fail
/// use control_rs_trace_macros::req;
///
/// #[req("VC-3.1")]
/// fn unqualified() {}
/// ```
///
/// A tag, `<doc>#<id>=<tag>`, is reserved and fails to compile:
///
/// ```compile_fail
/// use control_rs_trace_macros::req;
///
/// #[req("storage#VC-3.1=true")]
/// fn tagged() {}
/// ```
#[proc_macro_attribute]
pub fn req(args: TokenStream, item: TokenStream) -> TokenStream {
    match parse_ids(args.into()) {
        Ok(_) => item,
        Err(error) => {
            let mut out = error.to_compile_error();
            out.extend(TokenStream2::from(item));
            out.into()
        }
    }
}

/// The qualified IDs among the attribute arguments.
fn parse_ids(args: TokenStream2) -> Ids {
    let literals =
        Punctuated::<LitStr, Token![,]>::parse_terminated.parse2(args)?;
    if literals.is_empty() {
        return Err(syn::Error::new(
            Span::call_site(),
            "`#[req]` needs at least one qualified ID, such as \
             \"storage#VC-3.1\"",
        ));
    }
    literals
        .iter()
        .map(|literal| {
            let id = literal.value();
            if id.contains('=') {
                Err(syn::Error::new(
                    literal.span(),
                    format!("`{id}` carries a tag, reserved in this revision"),
                ))
            } else if is_qualified(&id) {
                Ok(id)
            } else {
                Err(syn::Error::new(
                    literal.span(),
                    format!("`{id}` is not a qualified ID `<doc>#<id>`"),
                ))
            }
        })
        .collect()
}

/// Whether `id` is `<doc>#<id>`: text on both sides of one `#`, without
/// whitespace.
fn is_qualified(id: &str) -> bool {
    id.split_once('#').is_some_and(|(doc, rest)| {
        !doc.is_empty() && !rest.is_empty() && !rest.contains('#')
    }) && !id.contains(char::is_whitespace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn qualified_ids_have_text_on_both_sides_of_one_hash() {
        assert!(is_qualified("storage#FR-3"));
        assert!(is_qualified("requirement-traceability#NFR-2"));
        for id in ["FR-3", "#FR-3", "storage#", "a#b#c", "a #FR-1", ""] {
            assert!(!is_qualified(id), "{id} is not qualified");
        }
    }

    #[test]
    fn arguments_are_string_literals_holding_qualified_ids() {
        let ids = parse_ids(quote!("a#FR-1", "b#C-2",)).unwrap();
        assert_eq!(ids, ["a#FR-1", "b#C-2"]);
        assert!(parse_ids(quote!()).is_err());
        assert!(parse_ids(quote!("FR-1")).is_err());
        assert!(parse_ids(quote!(a)).is_err());
        assert!(parse_ids(quote!("a#VC-1.1=true")).is_err());
    }
}
